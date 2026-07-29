"""
Indic LLM — Model Export & Format Conversion Bridge
=====================================================
Utilities for exporting trained IndicLLM checkpoints to:
  1. Hugging Face Transformers format (LlamaForCausalLM architecture)
     with full config.json, generation_config.json, and state_dict.
  2. PyTorch Dynamic INT8 Quantized weights for fast, low-memory CPU inference.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from model import ModelConfig
except ImportError:
    from src.model import ModelConfig  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export")


def convert_state_dict_to_hf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map IndicLLM parameter names to HuggingFace LlamaForCausalLM naming conventions.
    """
    hf_dict: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        # Embeddings & LM Head
        if k == "tok_embeddings.weight":
            hf_dict["model.embed_tokens.weight"] = v
        elif k == "output.weight":
            hf_dict["lm_head.weight"] = v
        elif k == "norm.weight":
            hf_dict["model.norm.weight"] = v

        # Transformer Layers
        elif k.startswith("layers."):
            parts = k.split(".")
            layer_idx = parts[1]
            submodule = ".".join(parts[2:])

            # Attention norms and FFN norms
            if submodule == "attn_norm.weight":
                hf_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = v
            elif submodule == "ffn_norm.weight":
                hf_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = v

            # Projections
            elif submodule == "attn.wq.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = v
            elif submodule == "attn.wk.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = v
            elif submodule == "attn.wv.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
            elif submodule == "attn.wo.weight":
                hf_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = v

            # SwiGLU FFN
            elif submodule == "ffn.w1.weight":
                hf_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = v
            elif submodule == "ffn.w2.weight":
                hf_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = v
            elif submodule == "ffn.w3.weight":
                hf_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = v
            else:
                # Retain any other internal layer buffer (e.g. KV cache)
                pass
        else:
            # Drop freqs_cis or runtime buffers
            pass

    return hf_dict


def build_hf_config(cfg: ModelConfig) -> Dict[str, Any]:
    """Generate Hugging Face LlamaConfig dictionary."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": cfg.pad_id,
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.ffn_hidden_dim,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "num_hidden_layers": cfg.n_layers,
        "rms_norm_eps": 1e-6,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_seq_len,
        "vocab_size": cfg.vocab_size,
        "model_type": "llama",
        "torch_dtype": "float32",
        "tie_word_embeddings": True,
        "sliding_window": cfg.sliding_window,
    }


def export_to_hf(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_path: Optional[str] = "data/tokenizer/indic_spm.model",
) -> str:
    """
    Export an IndicLLM checkpoint into Hugging Face Transformers directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info("Loading checkpoint for HF export: %s", checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_cfg = ckpt.get("config", {})
    model_cfg = ModelConfig(
        vocab_size=raw_cfg.get("vocab_size", 32000),
        dim=raw_cfg.get("model_dim", 512),
        n_layers=raw_cfg.get("n_layers", 8),
        n_heads=raw_cfg.get("n_heads", 8),
        n_kv_heads=raw_cfg.get("n_kv_heads", 4),
        max_seq_len=raw_cfg.get("max_seq_len", 512),
        dropout=0.0,
        sliding_window=raw_cfg.get("sliding_window", None),
    )

    # 1. Convert weights
    state_dict = ckpt["model_state_dict"]
    hf_state_dict = convert_state_dict_to_hf(state_dict)
    weights_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(hf_state_dict, weights_path)
    log.info("Saved HF state dict (%d tensors) -> %s", len(hf_state_dict), weights_path)

    # 2. Write config.json
    hf_config = build_hf_config(model_cfg)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    log.info("Saved config.json -> %s", config_path)

    # 3. Write generation_config.json
    gen_config = {
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": model_cfg.pad_id,
        "max_length": model_cfg.max_seq_len,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.92,
    }
    gen_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)

    # 4. Copy tokenizer model if available
    if tokenizer_path and os.path.exists(tokenizer_path):
        dest_tokenizer = os.path.join(output_dir, "tokenizer.model")
        shutil.copyfile(tokenizer_path, dest_tokenizer)
        log.info("Copied SentencePiece model -> %s", dest_tokenizer)

    log.info("✓ Hugging Face model exported successfully to: %s", output_dir)
    return output_dir

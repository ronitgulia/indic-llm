"""
Unit tests for model export and quantization bridge:
  - Hugging Face state dict key mapping
  - HF config.json generation
  - Dynamic INT8 quantization
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from export import build_hf_config, convert_state_dict_to_hf, quantize_model
from model import IndicLLM, ModelConfig


def test_convert_state_dict_to_hf():
    cfg = ModelConfig(vocab_size=100, dim=32, n_layers=2, n_heads=2, n_kv_heads=2, max_seq_len=64)
    model = IndicLLM(cfg)
    raw_sd = model.state_dict()

    hf_sd = convert_state_dict_to_hf(raw_sd)

    # Check key mappings
    assert "model.embed_tokens.weight" in hf_sd
    assert "lm_head.weight" in hf_sd
    assert "model.norm.weight" in hf_sd
    assert "model.layers.0.self_attn.q_proj.weight" in hf_sd
    assert "model.layers.0.self_attn.k_proj.weight" in hf_sd
    assert "model.layers.0.self_attn.v_proj.weight" in hf_sd
    assert "model.layers.0.self_attn.o_proj.weight" in hf_sd
    assert "model.layers.0.mlp.gate_proj.weight" in hf_sd
    assert "model.layers.0.mlp.down_proj.weight" in hf_sd
    assert "model.layers.0.mlp.up_proj.weight" in hf_sd


def test_build_hf_config():
    cfg = ModelConfig(vocab_size=32000, dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=512)
    hf_cfg = build_hf_config(cfg)

    assert hf_cfg["architectures"] == ["LlamaForCausalLM"]
    assert hf_cfg["hidden_size"] == 512
    assert hf_cfg["num_hidden_layers"] == 8
    assert hf_cfg["num_attention_heads"] == 8
    assert hf_cfg["num_key_value_heads"] == 4
    assert hf_cfg["vocab_size"] == 32000
    assert hf_cfg["model_type"] == "llama"


def test_quantize_model():
    cfg = ModelConfig(vocab_size=100, dim=32, n_layers=2, n_heads=2, n_kv_heads=2, max_seq_len=64)
    model = IndicLLM(cfg)

    quantized = quantize_model(model)
    # Check that linear layers are dynamically quantized
    assert hasattr(quantized.layers[0].attn.wq, "_packed_params") or "Dynamic" in str(type(quantized.layers[0].attn.wq))

    # Verify forward pass on quantized model
    x = torch.randint(0, 100, (1, 8))
    logits, _ = quantized(x)
    assert logits.shape == (1, 8, 100)

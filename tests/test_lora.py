"""
Unit tests for Low-Rank Adaptation (LoRA) module:
  - LoRALinear initialization and forward pass
  - Weight merging and unmerging equivalence
  - Model LoRA parameter injection and parameter freezing
  - Adapter weight checkpointing and deserialization
  - merge_and_unload fusion for inference
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from lora import (  # noqa: E402
    LoRAConfig,
    LoRALinear,
    apply_lora,
    load_adapter,
    merge_and_unload,
    save_adapter,
)
from model import IndicLLM, ModelConfig  # noqa: E402


def test_lora_linear_initialization():
    in_features, out_features, r = 32, 64, 4
    base_linear = nn.Linear(in_features, out_features, bias=False)
    lora = LoRALinear(base_linear, r=r, lora_alpha=8.0, lora_dropout=0.0)

    # Check shapes
    assert lora.lora_A.shape == (r, in_features)
    assert lora.lora_B.shape == (out_features, r)

    # Check base layer is frozen
    assert not lora.base_layer.weight.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad

    # lora_B is initialized to zeros -> delta W is exactly zero initially
    assert torch.all(lora.lora_B == 0.0)

    # Output initially matches base linear exactly
    x = torch.randn(2, 5, in_features)
    base_out = base_linear(x)
    lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_scaling_and_forward():
    in_features, out_features, r, alpha = 16, 16, 4, 8.0
    base = nn.Linear(in_features, out_features, bias=False)
    lora = LoRALinear(base, r=r, lora_alpha=alpha, lora_dropout=0.0)

    # Manually populate lora_B to test scaling
    with torch.no_grad():
        lora.lora_B.fill_(1.0)
        lora.lora_A.fill_(0.5)

    x = torch.ones(1, 1, in_features)
    expected_delta = (x @ lora.lora_A.t()) @ lora.lora_B.t() * (alpha / r)
    expected_out = base(x) + expected_delta

    assert torch.allclose(lora(x), expected_out, atol=1e-5)


def test_lora_merge_and_unmerge():
    in_features, out_features, r = 16, 16, 4
    base = nn.Linear(in_features, out_features, bias=False)
    orig_weight = base.weight.clone()

    lora = LoRALinear(base, r=r, lora_alpha=8.0, lora_dropout=0.0)
    with torch.no_grad():
        nn.init.normal_(lora.lora_B)

    x = torch.randn(2, 3, in_features)
    out_unmerged = lora(x)

    # Merge weights into base_layer
    lora.merge()
    assert lora.merged
    out_merged = lora(x)

    # Outputs must match before and after merge
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)
    assert not torch.allclose(base.weight, orig_weight)

    # Unmerge weights
    lora.unmerge()
    assert not lora.merged
    assert torch.allclose(base.weight, orig_weight, atol=1e-6)


def test_apply_lora_and_trainable_parameters():
    cfg = ModelConfig(
        vocab_size=100,
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=32,
    )
    model = IndicLLM(cfg)

    lora_cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["wq", "wv", "w1", "w2"])
    model = apply_lora(model, lora_cfg)

    # Verify target projections are LoRALinear
    for layer in model.layers:
        assert isinstance(layer.attn.wq, LoRALinear)
        assert isinstance(layer.attn.wv, LoRALinear)
        assert isinstance(layer.ffn.w1, LoRALinear)
        assert isinstance(layer.ffn.w2, LoRALinear)
        # Non-targeted should remain regular Linear
        assert isinstance(layer.attn.wo, nn.Linear)
        assert not isinstance(layer.attn.wo, LoRALinear)

    # Trainable parameters should be small fraction (< 10%)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable > 0
    assert trainable / total < 0.10


def test_save_and_load_adapter(tmp_path: Path):
    cfg = ModelConfig(
        vocab_size=50,
        dim=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=16,
    )
    model = IndicLLM(cfg)
    lora_cfg = LoRAConfig(r=2, lora_alpha=4.0, target_modules=["wq"])
    model = apply_lora(model, lora_cfg)

    # Perturb adapter weights
    with torch.no_grad():
        model.layers[0].attn.wq.lora_B.fill_(0.42)

    save_dir = str(tmp_path / "test_adapter")
    save_adapter(model, lora_cfg, save_dir)

    assert os.path.isfile(os.path.join(save_dir, "adapter_model.pt"))
    assert os.path.isfile(os.path.join(save_dir, "adapter_config.json"))

    # Create fresh model, apply lora, load weights
    fresh_model = IndicLLM(cfg)
    fresh_model = apply_lora(fresh_model, lora_cfg)
    load_adapter(fresh_model, save_dir)

    assert torch.allclose(
        fresh_model.layers[0].attn.wq.lora_B,
        model.layers[0].attn.wq.lora_B,
        atol=1e-6,
    )


def test_merge_and_unload():
    cfg = ModelConfig(
        vocab_size=50,
        dim=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=16,
        dropout=0.0,
    )
    model = IndicLLM(cfg)
    lora_cfg = LoRAConfig(r=2, lora_alpha=4.0, lora_dropout=0.0, target_modules=["wq", "w1"])
    model = apply_lora(model, lora_cfg)
    model.eval()

    x = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        out_before, _ = model(x)

    fused_model = merge_and_unload(model)
    fused_model.eval()

    # After merge_and_unload, modules should be standard nn.Linear
    assert isinstance(fused_model.layers[0].attn.wq, nn.Linear)
    assert not isinstance(fused_model.layers[0].attn.wq, LoRALinear)

    with torch.no_grad():
        out_after, _ = fused_model(x)

    assert torch.allclose(out_before, out_after, atol=1e-5)

"""
Unit tests for INT4 and INT8 Model Quantization:
  - WeightOnlyInt8Linear quantization scale and forward pass
  - WeightOnlyInt4Linear nibble packing and unpacking
  - In-place model weight quantization
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from model import IndicLLM, ModelConfig  # noqa: E402
from quantize import (  # noqa: E402
    WeightOnlyInt4Linear,
    WeightOnlyInt8Linear,
    quantize_model_weights,
)


def test_weight_only_int8():
    linear = nn.Linear(16, 32, bias=True)
    q8 = WeightOnlyInt8Linear.from_float(linear)

    assert q8.qweight.dtype == torch.int8
    assert q8.scale.shape == (32, 1)

    x = torch.randn(2, 4, 16)
    out_fp = linear(x)
    out_q8 = q8(x)
    sim = torch.cosine_similarity(out_fp.flatten(), out_q8.flatten(), dim=0)
    assert sim.item() > 0.98


def test_weight_only_int4():
    linear = nn.Linear(16, 16, bias=False)
    q4 = WeightOnlyInt4Linear.from_float(linear)

    assert q4.packed_weight.dtype == torch.uint8
    assert q4.packed_weight.shape == (16, 8)

    x = torch.randn(1, 2, 16)
    out_q4 = q4(x)
    assert out_q4.shape == (1, 2, 16)


def test_quantize_model_weights():
    cfg = ModelConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=16)
    model = IndicLLM(cfg)

    quantize_model_weights(model, bits=8)
    assert isinstance(model.layers[0].attn.wq, WeightOnlyInt8Linear)

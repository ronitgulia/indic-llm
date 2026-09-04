"""
Indic LLM — Block-Wise INT4 & INT8 Weight Quantization
======================================================
Implements post-training weight-only quantization for resource-constrained edge devices:
  - WeightOnlyInt8Linear : Squeezes weights to signed 8-bit integers (50% memory saving)
  - WeightOnlyInt4Linear : Packs weights into 4-bit nibbles (75% memory saving)
  - quantize_model_weights: In-place conversion of model linear projections
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
sys_path = str(repo_root / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from model import IndicLLM
except ImportError:
    from src.model import IndicLLM  # type: ignore[no-redef]

log = logging.getLogger("quantize")


class WeightOnlyInt8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> WeightOnlyInt8Linear:
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            w = linear.weight.float()
            scale = w.abs().amax(dim=1, keepdim=True) / 127.0
            scale = torch.clamp(scale, min=1e-8)
            qweight = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)

            layer.qweight.copy_(qweight)
            layer.scale.copy_(scale)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.float())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp = self.qweight.float() * self.scale
        return torch.nn.functional.linear(x, w_fp, self.bias)

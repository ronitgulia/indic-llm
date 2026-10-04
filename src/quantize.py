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


class WeightOnlyInt4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        packed_in = math.ceil(in_features / 2)
        self.register_buffer("packed_weight", torch.zeros((out_features, packed_in), dtype=torch.uint8))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> WeightOnlyInt4Linear:
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            w = linear.weight.float()
            scale = w.abs().amax(dim=1, keepdim=True) / 7.0
            scale = torch.clamp(scale, min=1e-8)
            q_signed = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)
            q_unsigned = (q_signed + 8).to(torch.uint8)

            even = q_unsigned[:, 0::2]
            odd = q_unsigned[:, 1::2] if linear.in_features % 2 == 0 else torch.zeros_like(even)
            packed = (even & 0x0F) | ((odd & 0x0F) << 4)

            layer.packed_weight.copy_(packed)
            layer.scale.copy_(scale)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.float())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even = (self.packed_weight & 0x0F).to(torch.int8) - 8
        odd = ((self.packed_weight >> 4) & 0x0F).to(torch.int8) - 8

        unpacked = torch.empty((self.out_features, self.in_features), dtype=torch.float32, device=x.device)
        unpacked[:, 0::2] = even[:, : unpacked[:, 0::2].shape[1]].float()
        if self.in_features % 2 == 0:
            unpacked[:, 1::2] = odd.float()

        w_fp = unpacked * self.scale
        return torch.nn.functional.linear(x, w_fp, self.bias)


def quantize_model_weights(model: IndicLLM, bits: int = 8) -> IndicLLM:
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bit width: {bits}. Must be 4 or 8.")

    quant_cls = WeightOnlyInt8Linear if bits == 8 else WeightOnlyInt4Linear
    converted = 0

    for layer in model.layers:
        for container in (layer.attn, layer.ffn):
            for attr in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
                if hasattr(container, attr):
                    orig_mod = getattr(container, attr)
                    if isinstance(orig_mod, nn.Linear):
                        q_mod = quant_cls.from_float(orig_mod)
                        setattr(container, attr, q_mod)
                        converted += 1

    log.info("Quantized %d linear layers to INT%d", converted, bits)
    return model

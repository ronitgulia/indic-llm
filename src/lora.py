"""
Indic LLM — Low-Rank Adaptation (LoRA) PEFT Module
===================================================
Parameter-Efficient Fine-Tuning using Low-Rank Adaptation (Hu et al., 2021).
Enables efficient adaptation of IndicLLM models with < 1% trainable parameters:

  - LoRALinear: Drop-in replacement for nn.Linear with trainable low-rank matrices
  - apply_lora: Injects LoRA adapters into target attention & MLP projections
  - merge_and_unload: Fuses adapter weights back into base linear layers for zero-overhead inference
  - save_adapter / load_adapter: Lightweight adapter checkpoint management
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Ensure src/ is importable
sys_path = str(Path(__file__).resolve().parent)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from model import IndicLLM
except ImportError:
    from src.model import IndicLLM  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lora")


@dataclass
class LoRAConfig:
    """Hyperparameters for Low-Rank Adaptation."""

    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["wq", "wv", "w1", "w2"]
    )


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with low-rank adaptation matrices:
      h = W_0 x + (B @ A) x * (alpha / r)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Initialize low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        lora_in = self.lora_dropout(x)
        # (B, T, in_features) @ (in_features, r) -> (B, T, r)
        # (B, T, r) @ (r, out_features) -> (B, T, out_features)
        lora_out = (lora_in @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Fuse adapter weights into base_layer.weight in-place."""
        if not self.merged:
            weight_delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data.add_(weight_delta)
            self.merged = True

    def unmerge(self) -> None:
        """Remove fused adapter weights from base_layer.weight."""
        if self.merged:
            weight_delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data.sub_(weight_delta)
            self.merged = False


def apply_lora(model: IndicLLM, config: Optional[LoRAConfig] = None) -> IndicLLM:
    """
    Inject LoRA adapters into target layers of IndicLLM and freeze base weights.
    """
    cfg = config or LoRAConfig()

    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False

    total_injected = 0
    for layer_idx, layer in enumerate(model.layers):
        for name in cfg.target_modules:
            # Check attention modules
            if hasattr(layer.attn, name):
                orig_proj = getattr(layer.attn, name)
                if isinstance(orig_proj, nn.Linear):
                    lora_proj = LoRALinear(
                        orig_proj,
                        r=cfg.r,
                        lora_alpha=cfg.lora_alpha,
                        lora_dropout=cfg.lora_dropout,
                    )
                    setattr(layer.attn, name, lora_proj)
                    total_injected += 1

            # Check FFN modules
            elif hasattr(layer.ffn, name):
                orig_proj = getattr(layer.ffn, name)
                if isinstance(orig_proj, nn.Linear):
                    lora_proj = LoRALinear(
                        orig_proj,
                        r=cfg.r,
                        lora_alpha=cfg.lora_alpha,
                        lora_dropout=cfg.lora_dropout,
                    )
                    setattr(layer.ffn, name, lora_proj)
                    total_injected += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable_params / max(1, total_params)

    log.info(
        "LoRA injected into %d layers | Trainable: %d / %d (%.2f%%)",
        total_injected,
        trainable_params,
        total_params,
        pct,
    )
    return model


def save_adapter(model: IndicLLM, config: LoRAConfig, output_dir: str) -> str:
    """Save only the trained LoRA adapter weights and configuration."""
    os.makedirs(output_dir, exist_ok=True)
    adapter_weights: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapter_weights[f"{name}.lora_A"] = module.lora_A.data.cpu()
            adapter_weights[f"{name}.lora_B"] = module.lora_B.data.cpu()

    weights_path = os.path.join(output_dir, "adapter_model.pt")
    torch.save(adapter_weights, weights_path)

    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    log.info("Saved %d LoRA tensors -> %s", len(adapter_weights), weights_path)
    return output_dir


def load_adapter(model: IndicLLM, adapter_dir: str) -> IndicLLM:
    """
    Load adapter weights from adapter_model.pt into an IndicLLM model.
    """
    weights_path = os.path.join(adapter_dir, "adapter_model.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Adapter weights not found at: {weights_path}")

    device = next(model.parameters()).device
    adapter_weights = torch.load(weights_path, map_location=device)
    model_modules = dict(model.named_modules())

    loaded_count = 0
    for key, weight in adapter_weights.items():
        mod_name, param_name = key.rsplit(".", 1)
        if mod_name in model_modules:
            mod = model_modules[mod_name]
            if hasattr(mod, param_name):
                param = getattr(mod, param_name)
                with torch.no_grad():
                    param.copy_(weight.to(param.device))
                loaded_count += 1

    log.info("Loaded %d LoRA weight tensors from %s", loaded_count, weights_path)
    return model


def merge_and_unload(model: IndicLLM) -> IndicLLM:
    """
    Merge all LoRA adapters into base weights and replace LoRALinear with pure nn.Linear.
    Returns the fused model ready for standard inference.
    """
    for layer in model.layers:
        for module_container in (layer.attn, layer.ffn):
            for attr_name in list(module_container.__dict__.keys()) + [
                "wq", "wk", "wv", "wo", "w1", "w2", "w3"
            ]:
                if hasattr(module_container, attr_name):
                    module = getattr(module_container, attr_name)
                    if isinstance(module, LoRALinear):
                        module.merge()
                        setattr(module_container, attr_name, module.base_layer)

    log.info("Merged all LoRA adapters into base model weights")
    return model

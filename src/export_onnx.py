"""
Indic LLM — ONNX Export & Runtime Acceleration
===============================================
Exports IndicLLM models to ONNX format with dynamic batching and sequence axes:
  - export_to_onnx    : Converts PyTorch state_dict into optimized ONNX graph
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

repo_root = Path(__file__).resolve().parent.parent
sys_path = str(repo_root / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]

log = logging.getLogger("export_onnx")


def export_to_onnx(
    model: IndicLLM,
    output_path: str,
    opset_version: int = 17,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.long, device=device)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }

    log.info("Exporting IndicLLM to ONNX: %s (opset %d)", output_path, opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    log.info("Successfully exported ONNX model -> %s", output_path)
    return output_path

"""
Indic LLM — Speculative Decoding Engine
========================================
Accelerates autoregressive generation by using a small draft model to propose
candidate tokens that are verified in parallel by the target model in a single
forward pass (Leviathan et al., 2023; Chen et al., 2023).

  - SpeculativeConfig  : Hyperparameters for draft lookahead and sampling
  - SpeculativeMetrics : Throughput, acceptance rate, and latency tracking
  - SpeculativeEngine  : Manages draft proposal, parallel verification, and residual correction
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
sys_path = str(repo_root / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("speculative")


@dataclass
class SpeculativeConfig:
    draft_steps: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: 150
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class SpeculativeMetrics:
    tokens_generated: int
    draft_tokens_proposed: int
    accepted_tokens: int
    target_forward_passes: int
    acceptance_rate: float
    total_time_sec: float
    tokens_per_second: float

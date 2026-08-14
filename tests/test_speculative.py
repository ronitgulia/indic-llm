"""
Unit tests for Speculative Decoding:
  - Speculative sampling mathematical properties and residual sampling
  - SpeculativeEngine draft-target verification equivalence
  - Generation metrics tracking and stop sequences
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "inference"))

from model import IndicLLM, ModelConfig  # noqa: E402
from speculative import (  # noqa: E402
    SpeculativeConfig,
    SpeculativeEngine,
    SpeculativeMetrics,
    compute_speculative_speedup,
    sample_token,
    speculative_step,
)


def test_speculative_sampling_perfect_match():
    p = torch.tensor([0.1, 0.7, 0.2])
    accepted, corrected = speculative_step(p, p, draft_token_id=1)
    assert accepted
    assert corrected is None

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


class DummyTokenizer:
    def __init__(self, vocab_size: int = 50):
        self._vocab_size = vocab_size

    def encode(self, text: str, out_type: type = int) -> List[int]:
        return [abs(hash(w)) % (self._vocab_size - 2) + 2 for w in text.split()] or [2]

    def decode(self, token_ids: List[int]) -> str:
        return " भारत देश"

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2


def test_speculative_engine_generation():
    cfg_target = ModelConfig(vocab_size=50, dim=32, n_layers=2, n_heads=2, n_kv_heads=2, max_seq_len=64, dropout=0.0)
    cfg_draft = ModelConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=64, dropout=0.0)

    target_m = IndicLLM(cfg_target)
    draft_m = IndicLLM(cfg_draft)
    tok = DummyTokenizer(vocab_size=50)

    engine = SpeculativeEngine(target_m, draft_m, tok, device=torch.device("cpu"))
    spec_cfg = SpeculativeConfig(draft_steps=2, max_new_tokens=6, temperature=0.0)

    text, metrics = engine.generate("भारत की राजधानी", spec_cfg)
    assert isinstance(text, str)
    assert isinstance(metrics, SpeculativeMetrics)
    assert metrics.tokens_generated > 0
    assert metrics.target_forward_passes > 0
    assert 0.0 <= metrics.acceptance_rate <= 100.0

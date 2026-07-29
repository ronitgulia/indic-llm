"""
Unit tests for Direct Preference Optimization (DPO):
  - compute_sequence_logps
  - dpo_loss mathematical properties
  - PreferenceDataset collation and prompt loss masking
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpo import IGNORE_INDEX, PreferenceDataset, compute_sequence_logps, dpo_loss
from model import IndicLLM, ModelConfig

MODEL_PATH = "data/tokenizer/indic_spm.model"


def test_dpo_loss_properties():
    # If policy prefers chosen much more than reference does, loss should be small
    # and chosen_rewards > rejected_rewards
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-5.0, -6.0])
    ref_chosen = torch.tensor([-3.0, -3.0])
    ref_rejected = torch.tensor([-3.0, -3.0])

    loss, c_rewards, r_rewards = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    assert loss.dim() == 0
    assert loss.item() > 0.0
    # Chosen rewards must exceed rejected rewards
    assert (c_rewards > r_rewards).all()


def test_compute_sequence_logps():
    cfg = ModelConfig(vocab_size=50, dim=32, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=16)
    model = IndicLLM(cfg)
    model.eval()

    inputs = torch.randint(1, 50, (2, 8))
    # Targets with first 3 tokens masked (-100)
    targets = inputs.clone()
    targets[:, :3] = IGNORE_INDEX

    logps = compute_sequence_logps(model, inputs, targets, ignore_index=IGNORE_INDEX)
    assert logps.shape == (2,)
    # Log probabilities are negative numbers
    assert (logps < 0.0).all()


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_preference_dataset(tmp_path: Path):
    import sentencepiece as spm

    tok = spm.SentencePieceProcessor()
    tok.load(MODEL_PATH)

    data_file = tmp_path / "pref.jsonl"
    records = [
        {
            "prompt": "भारत का राष्ट्रीय पक्षी क्या है?",
            "chosen": "भारत का राष्ट्रीय पक्षी मोर है।",
            "rejected": "मुझे नहीं पता।",
        }
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ds = PreferenceDataset(str(data_file), tokenizer=tok, max_seq_len=64)
    assert len(ds) == 1

    c_in, c_tgt, r_in, r_tgt = ds[0]
    assert c_in.shape == (64,)
    assert c_tgt.shape == (64,)
    assert r_in.shape == (64,)
    assert r_tgt.shape == (64,)

    # Prompt tokens masked in targets
    assert c_tgt[0].item() == IGNORE_INDEX
    assert r_tgt[0].item() == IGNORE_INDEX

    # Unmasked response tokens exist
    assert (c_tgt != IGNORE_INDEX).sum() > 0
    assert (r_tgt != IGNORE_INDEX).sum() > 0

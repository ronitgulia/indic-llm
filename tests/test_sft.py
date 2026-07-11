"""
Unit tests for Supervised Fine-Tuning (SFT) module:
  - InstructDataset record parsing
  - Target-only loss masking with IGNORE_INDEX (-100)
  - End-to-end forward loss computation with masked targets
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import IndicLLM, ModelConfig
from sft import IGNORE_INDEX, InstructDataset

MODEL_PATH = "data/tokenizer/indic_spm.model"


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    data_file = tmp_path / "instruct_sample.jsonl"
    records = [
        {"instruction": "भारत की राजधानी क्या है?", "output": "भारत की राजधानी नई दिल्ली है।"},
        {"instruction": "सूर्य किस दिशा से उगता है?", "output": "सूर्य पूर्व दिशा से उगता है।"},
        {
            "messages": [
                {"role": "user", "content": "पेड़ लगाने के क्या फायदे हैं?"},
                {"role": "assistant", "content": "पेड़ लगाने से पर्यावरण स्वच्छ रहता है और ऑक्सीजन मिलती है।"},
            ]
        },
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return data_file


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_instruct_dataset_parsing(sample_jsonl: Path):
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(MODEL_PATH)

    dataset = InstructDataset(str(sample_jsonl), tokenizer=tokenizer, max_seq_len=64)
    assert len(dataset) == 3


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_prompt_loss_masking(sample_jsonl: Path):
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(MODEL_PATH)

    dataset = InstructDataset(str(sample_jsonl), tokenizer=tokenizer, max_seq_len=64)
    inputs, targets = dataset[0]

    assert inputs.shape == (64,)
    assert targets.shape == (64,)

    # The beginning of targets should be masked (-100) for the prompt
    assert targets[0].item() == IGNORE_INDEX

    # There must be valid response targets that are NOT -100
    valid_targets = targets[targets != IGNORE_INDEX]
    assert len(valid_targets) > 0

    # Ensure padding tokens in targets are also -100
    assert targets[-1].item() == IGNORE_INDEX


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_sft_loss_computation(sample_jsonl: Path):
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(MODEL_PATH)

    cfg = ModelConfig(
        vocab_size=tokenizer.get_piece_size(),
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
    )
    model = IndicLLM(cfg)
    model.train()

    dataset = InstructDataset(str(sample_jsonl), tokenizer=tokenizer, max_seq_len=64)
    inputs, targets = dataset[0]

    batch_inputs = inputs.unsqueeze(0)
    batch_targets = targets.unsqueeze(0)

    logits, loss = model(batch_inputs, targets=batch_targets, ignore_index=IGNORE_INDEX)
    assert loss is not None
    assert loss.item() > 0.0

    # Verify gradients flow backward cleanly
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None

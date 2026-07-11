"""
Unit tests for IndicTokenizer and SentencePiece BPE:
  - Vocabulary properties and special tokens
  - Encode / decode round-trip integrity on Devanagari text
  - Preserving matras, nuktas, halants, and conjuncts
  - Batch encoding and padding
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizer import IndicTokenizer

MODEL_PATH = "data/tokenizer/indic_spm.model"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_tokenizer_properties():
    tok = IndicTokenizer(MODEL_PATH)
    assert tok.vocab_size > 0
    assert tok.bos_id == 2
    assert tok.eos_id == 3
    assert tok.pad_id == 0


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_devanagari_roundtrip():
    tok = IndicTokenizer(MODEL_PATH)

    phrases = [
        "नमस्ते दुनिया!",
        "भारत एक महान और लोकतांत्रिक देश है।",
        "विज्ञान और प्रौद्योगिकी का विकास",
        "ऋषि, ज्ञान, क्षत्रिय, और त्रिशूल",  # Complex conjuncts and matras
    ]

    for phrase in phrases:
        ids = tok.encode(phrase, add_bos=False, add_eos=False)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert decoded.strip() == phrase.strip()


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_special_tokens():
    tok = IndicTokenizer(MODEL_PATH)

    text = "नमशकार"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Tokenizer model not found on disk")
def test_batch_encode():
    tok = IndicTokenizer(MODEL_PATH)

    batch = ["पहला वाक्य", "यह दूसरा और थोड़ा लंबा वाक्य है"]
    encoded = tok.encode_batch(batch, pad_to_length=16)

    assert len(encoded) == 2
    assert len(encoded[0]) == 16
    assert len(encoded[1]) == 16
    # Shorter sequence has padding token (0)
    assert encoded[0][-1] == tok.pad_id

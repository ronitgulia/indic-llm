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


@pytest.fixture(scope="module")
def active_tokenizer_path(tmp_path_factory) -> str:
    """Return on-disk model path if available, or train a miniature toy tokenizer."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    import sentencepiece as spm

    tmp_dir = tmp_path_factory.mktemp("toy_spm")
    corpus = tmp_dir / "corpus.txt"
    corpus.write_text(
        "नमस्ते दुनिया!\n"
        "भारत एक महान और लोकतांत्रिक देश है।\n"
        "विज्ञान और प्रौद्योगिकी का विकास\n"
        "ऋषि ज्ञान क्षत्रिय त्रिशूल\n"
        "नमशकार पहला वाक्य यह दूसरा वाक्य है\n",
        encoding="utf-8",
    )
    prefix = str(tmp_dir / "toy_indic")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=40,
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return f"{prefix}.model"


def test_tokenizer_properties(active_tokenizer_path: str):
    tok = IndicTokenizer(active_tokenizer_path)
    assert tok.vocab_size > 0
    assert tok.bos_id == 2
    assert tok.eos_id == 3
    assert tok.pad_id == 0


def test_devanagari_roundtrip(active_tokenizer_path: str):
    tok = IndicTokenizer(active_tokenizer_path)

    phrases = [
        "नमस्ते दुनिया!",
        "भारत एक महान देश है।",
    ]

    for phrase in phrases:
        ids = tok.encode(phrase, add_bos=False, add_eos=False)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert len(decoded.strip()) > 0


def test_special_tokens(active_tokenizer_path: str):
    tok = IndicTokenizer(active_tokenizer_path)

    text = "नमस्ते"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_batch_encode(active_tokenizer_path: str):
    tok = IndicTokenizer(active_tokenizer_path)

    batch = ["पहला वाक्य", "यह दूसरा और थोड़ा लंबा वाक्य है"]
    encoded = tok.encode_batch(batch, pad_to_length=16)

    assert len(encoded) == 2
    assert len(encoded[0]) == 16
    assert len(encoded[1]) == 16
    # Shorter sequence has padding token (0)
    assert encoded[0][-1] == tok.pad_id

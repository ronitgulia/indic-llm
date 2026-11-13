"""
Unit tests for high-performance streaming inference engine:
  - GenerationConfig options
  - InferenceEngine streaming token generation with KV-cache
  - Multi-token stop sequence truncation
  - Non-streaming generate wrapper
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure src/ and inference/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from engine import GenerationConfig, InferenceEngine

from model import IndicLLM, ModelConfig


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [10, 20, 30]
    tok.decode.side_effect = lambda ids: f"tok_{ids[0]} "
    tok.eos_id.return_value = 999
    tok.bos_id.return_value = 1
    return tok


@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        vocab_size=100,
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
    )
    model = IndicLLM(cfg)
    model.eval()
    return model


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 256
    assert cfg.temperature == 0.8
    assert cfg.top_p == 0.92
    assert cfg.repetition_penalty == 1.15
    assert cfg.stop_sequences == []


def test_inference_engine_streaming(tiny_model, mock_tokenizer):
    engine = InferenceEngine(tiny_model, mock_tokenizer, device=torch.device("cpu"))
    config = GenerationConfig(max_new_tokens=4, temperature=0.5)

    tokens_received = []
    rates = []

    for token_text, tok_s in engine.generate_stream("नमस्ते", config):
        tokens_received.append(token_text)
        rates.append(tok_s)

    assert len(tokens_received) == 4
    assert len(rates) == 4
    assert all(r >= 0.0 for r in rates)


def test_inference_engine_stop_sequence(tiny_model, mock_tokenizer):
    mock_tokenizer.decode.side_effect = ["word ", "STOP", "extra "]
    engine = InferenceEngine(tiny_model, mock_tokenizer, device=torch.device("cpu"))
    config = GenerationConfig(max_new_tokens=10, stop_sequences=["STOP"])

    tokens_received = []
    for token_text, _ in engine.generate_stream("परीक्षण", config):
        tokens_received.append(token_text)

    # Should stop when hitting "STOP"
    assert "extra " not in tokens_received


def test_inference_engine_generate_wrapper(tiny_model, mock_tokenizer):
    engine = InferenceEngine(tiny_model, mock_tokenizer, device=torch.device("cpu"))
    config = GenerationConfig(max_new_tokens=3)

    text, tok_s = engine.generate("भारत", config)
    assert isinstance(text, str)
    assert len(text) > 0
    assert tok_s >= 0.0


def test_inference_engine_greedy_decoding(tiny_model, mock_tokenizer):
    engine = InferenceEngine(tiny_model, mock_tokenizer, device=torch.device("cpu"))
    config = GenerationConfig(max_new_tokens=4, temperature=0.0)

    tokens = [tok for tok, _ in engine.generate_stream("नमस्ते", config)]
    assert len(tokens) == 4


def test_inference_engine_stop_sequence_trimmed_in_generate(tiny_model, mock_tokenizer):
    mock_tokenizer.decode.side_effect = ["नमस्ते ", "दुनिया ", "<|stop|>", "अतिरिक्त "]
    engine = InferenceEngine(tiny_model, mock_tokenizer, device=torch.device("cpu"))
    config = GenerationConfig(max_new_tokens=6, stop_sequences=["<|stop|>"])

    text, _ = engine.generate("जांच", config)
    assert "<|stop|>" not in text
    assert "अतिरिक्त " not in text

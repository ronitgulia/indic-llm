"""
Unit tests for Indic LLM Benchmark Suite:
  - MCQASample dataclass and JSONL dataset loading
  - Prompt formatting variants
  - Choice log-probability computation (compute_choice_logp)
  - Full MCQA evaluation engine (evaluate_mcqa)
  - Markdown report and multi-model leaderboard generation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest
import torch

# Ensure repo root and src/ are importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "eval"))

from benchmark_suite import (  # noqa: E402
    BenchmarkResult,
    MCQASample,
    PredictionResult,
    compute_choice_logp,
    evaluate_mcqa,
    format_mcqa_prompt,
    generate_leaderboard_table,
    generate_markdown_report,
    load_mcqa_dataset,
)

from model import IndicLLM, ModelConfig  # noqa: E402


class DummyTokenizer:
    """Mock SentencePiece tokenizer for deterministic unit testing."""

    def __init__(self, vocab_size: int = 100):
        self._vocab_size = vocab_size

    def encode(self, text: str, out_type: type = int) -> List[int]:
        tokens = [abs(hash(tok)) % (self._vocab_size - 2) + 2 for tok in text.split()]
        return tokens or [2]

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def pad_id(self) -> int:
        return 0


@pytest.fixture
def dummy_model() -> IndicLLM:
    cfg = ModelConfig(
        vocab_size=100,
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
    )
    model = IndicLLM(cfg)
    model.eval()
    return model


@pytest.fixture
def sample_mcqa_file(tmp_path: Path) -> Path:
    samples = [
        {
            "question": "भारत की राजधानी क्या है?",
            "options": ["मुंबई", "नई दिल्ली", "कोलकाता", "चेन्नई"],
            "answer_idx": 1,
            "category": "General Knowledge",
        },
        {
            "question": "सूर्य किस दिशा से निकलता है?",
            "options": ["पश्चिम", "उत्तर", "दक्षिण", "पूर्व"],
            "answer_idx": 3,
            "category": "Common Sense",
        },
        {
            "question": "'कमल' का पर्यायवाची क्या है?",
            "options": ["जलज", "अग्नि", "पवन", "भूमि"],
            "answer_idx": 0,
            "category": "Grammar",
            "context": "पर्यायवाची शब्द समान अर्थ प्रकट करते हैं।",
        },
    ]
    file_path = tmp_path / "test_mcqa.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        # Add an invalid line to verify robust skipping
        f.write("invalid json line\n")
    return file_path


def test_load_mcqa_dataset(sample_mcqa_file: Path):
    samples = load_mcqa_dataset(sample_mcqa_file)
    assert len(samples) == 3

    s1 = samples[0]
    assert s1.question == "भारत की राजधानी क्या है?"
    assert s1.options == ["मुंबई", "नई दिल्ली", "कोलकाता", "चेन्नई"]
    assert s1.answer_idx == 1
    assert s1.category == "General Knowledge"

    s3 = samples[2]
    assert s3.context == "पर्यायवाची शब्द समान अर्थ प्रकट करते हैं।"


def test_format_mcqa_prompt():
    sample = MCQASample(
        question="प्रश्न?",
        options=["A", "B"],
        answer_idx=0,
        category="Test",
        context="संदर्भ पाठ",
    )
    direct_prompt = format_mcqa_prompt(sample, style="direct")
    assert direct_prompt == "प्रश्न: प्रश्न?\nउत्तर:"

    context_prompt = format_mcqa_prompt(sample, style="context")
    assert context_prompt == "संदर्भ: संदर्भ पाठ\nप्रश्न: प्रश्न?\nउत्तर:"


def test_compute_choice_logp(dummy_model: IndicLLM):
    tok = DummyTokenizer()
    prompt = "प्रश्न: भारत की राजधानी क्या है?\nउत्तर:"
    choice = "नई दिल्ली"

    logp_norm = compute_choice_logp(
        model=dummy_model,
        tokenizer=tok,  # type: ignore[arg-type]
        prompt=prompt,
        choice=choice,
        device=torch.device("cpu"),
        length_norm=True,
    )

    logp_raw = compute_choice_logp(
        model=dummy_model,
        tokenizer=tok,  # type: ignore[arg-type]
        prompt=prompt,
        choice=choice,
        device=torch.device("cpu"),
        length_norm=False,
    )

    assert isinstance(logp_norm, float)
    assert isinstance(logp_raw, float)
    assert logp_norm <= 0.0
    assert logp_raw <= 0.0
    # Raw logp has magnitude >= normalized logp
    assert abs(logp_raw) >= abs(logp_norm) - 1e-5


def test_evaluate_mcqa(dummy_model: IndicLLM, sample_mcqa_file: Path):
    tok = DummyTokenizer()
    result = evaluate_mcqa(
        model=dummy_model,
        tokenizer=tok,  # type: ignore[arg-type]
        samples_or_path=sample_mcqa_file,
        device=torch.device("cpu"),
    )

    assert isinstance(result, BenchmarkResult)
    assert result.total_samples == 3
    assert 0 <= result.correct_samples <= 3
    assert 0.0 <= result.accuracy <= 100.0

    # Check categories breakdown
    cats = result.category_metrics
    assert "General Knowledge" in cats
    assert "Common Sense" in cats
    assert "Grammar" in cats
    assert cats["General Knowledge"]["total"] == 1

    # Check predictions details
    assert len(result.predictions) == 3
    p1 = result.predictions[0]
    assert isinstance(p1, PredictionResult)
    assert len(p1.choice_scores) == 4
    assert 0 <= p1.predicted_idx < 4


def test_generate_markdown_report(tmp_path: Path):
    result = BenchmarkResult(
        total_samples=2,
        correct_samples=2,
        accuracy=100.0,
        category_metrics={
            "Grammar": {"correct": 1, "total": 1, "accuracy": 100.0},
            "History": {"correct": 1, "total": 1, "accuracy": 100.0},
        },
        predictions=[
            PredictionResult(
                question="Q1?",
                category="Grammar",
                options=["A", "B"],
                correct_idx=0,
                predicted_idx=0,
                choice_scores=[-1.0, -2.5],
                is_correct=True,
            )
        ],
    )

    out_file = tmp_path / "test_report.md"
    content = generate_markdown_report(result, model_name="TestModel", output_path=out_file)

    assert out_file.is_file()
    assert "# Indic LLM — Benchmark Evaluation Report" in content
    assert "TestModel" in content
    assert "100.00%" in content
    assert "Grammar" in content
    assert "History" in content


def test_generate_leaderboard_table(tmp_path: Path):
    res_base = BenchmarkResult(
        total_samples=4,
        correct_samples=2,
        accuracy=50.0,
        category_metrics={
            "Grammar": {"correct": 1, "total": 2, "accuracy": 50.0},
            "Reasoning": {"correct": 1, "total": 2, "accuracy": 50.0},
        },
    )
    res_lora = BenchmarkResult(
        total_samples=4,
        correct_samples=4,
        accuracy=100.0,
        category_metrics={
            "Grammar": {"correct": 2, "total": 2, "accuracy": 100.0},
            "Reasoning": {"correct": 2, "total": 2, "accuracy": 100.0},
        },
    )

    leaderboard_map = {
        "IndicLLM-Base": res_base,
        "IndicLLM-LoRA": res_lora,
    }

    out_file = tmp_path / "leaderboard.md"
    table = generate_leaderboard_table(leaderboard_map, output_path=out_file)

    assert out_file.is_file()
    assert "Indic LLM Benchmark Leaderboard" in table
    assert "IndicLLM-Base" in table
    assert "IndicLLM-LoRA" in table
    assert "50.0%" in table
    assert "100.0%" in table

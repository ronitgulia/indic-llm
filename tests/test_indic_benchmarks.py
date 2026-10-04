"""Unit tests for Indic BLEU, ChrF++, and extended evaluation metrics."""

import pytest

from eval.indic_benchmarks import compute_bleu, compute_chrf, tokenize_indic


def test_tokenize_indic():
    tokens = tokenize_indic("नमस्ते, दुनिया! आप कैसे हैं?")
    assert "नमस्ते" in tokens
    assert "दुनिया" in tokens
    assert "," in tokens
    assert "?" in tokens


def test_compute_bleu_identical():
    hyps = ["भारत एक विशाल और सुंदर देश है।"]
    refs = [["भारत एक विशाल और सुंदर देश है।"]]
    score = compute_bleu(hyps, refs)
    assert score >= 99.0


def test_compute_bleu_partial():
    hyps = ["भारत एक विशाल देश है।"]
    refs = [["भारत एक सुंदर और विशाल देश है।"]]
    score = compute_bleu(hyps, refs)
    assert 40.0 < score < 100.0


def test_compute_chrf():
    hyps = ["सौर ऊर्जा पर्यावरण के अनुकूल है।"]
    refs = [["सौर ऊर्जा पर्यावरण के अनुकूल है।"]]
    score = compute_chrf(hyps, refs)
    assert score >= 99.0


def test_reasoning_benchmark():
    from eval.indic_benchmarks import IndicReasoningBenchmark
    bench = IndicReasoningBenchmark()
    preds = [
        {"predicted_answer": "9 सेब बचे", "ground_truth": "9"},
        {"predicted_answer": "6", "ground_truth": "6"},
        {"predicted_answer": "गलत उत्तर", "ground_truth": "17"},
    ]
    res = bench.evaluate_predictions(preds)
    assert res.score == pytest.approx(66.67, 0.1)
    assert res.num_samples == 3


def test_transliteration_accuracy():
    from eval.indic_benchmarks import TransliterationAccuracyMetric
    hyps = ["नमस्ते", "भारत", "गलत"]
    refs = ["नमस्ते", "भारत", "सही"]
    acc = TransliterationAccuracyMetric.compute_accuracy(hyps, refs)
    assert acc == pytest.approx(66.67, 0.1)


def test_cross_lingual_transfer():
    from eval.indic_benchmarks import CrossLingualTransferScore
    src = [80.0, 90.0]
    tgt = [72.0, 81.0]
    parity = CrossLingualTransferScore.compute_parity(src, tgt)
    assert parity == 90.0

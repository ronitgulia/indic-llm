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

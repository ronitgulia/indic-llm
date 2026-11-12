"""
Unit tests for lightweight evaluation benchmarks:
  - compute_bleu()
  - compute_chrf()
  - token_accuracy()
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure eval/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from benchmarks import compute_bleu, compute_chrf, token_accuracy


def test_bleu_identical_strings():
    refs = ["यह एक सुंदर दिन है", "भारत एक विशाल देश है"]
    hyps = ["यह एक सुंदर दिन है", "भारत एक विशाल देश है"]
    result = compute_bleu(refs, hyps)

    assert result["bleu"] > 0.99
    assert result["brevity_penalty"] == 1.0
    for p in result["precisions"]:
        assert p > 0.99


def test_bleu_short_identical_strings():
    # 2-word identical strings should not collapse to 0.0 due to max_order=4
    refs = ["नमस्ते भारत"]
    hyps = ["नमस्ते भारत"]
    result = compute_bleu(refs, hyps)

    assert result["bleu"] > 0.99
    assert result["brevity_penalty"] == 1.0
    assert len(result["precisions"]) == 2
    for p in result["precisions"]:
        assert p > 0.99


def test_bleu_single_token():
    refs = ["नमस्ते"]
    hyps = ["नमस्ते"]
    result = compute_bleu(refs, hyps)

    assert result["bleu"] == 1.0
    assert result["precisions"] == [1.0]


def test_bleu_empty_inputs():
    result = compute_bleu([], [])
    assert result["bleu"] == 0.0
    assert result["hyp_len"] == 0


def test_bleu_disjoint_strings():
    refs = ["सूरज पूरब में उगता है"]
    hyps = ["चांद रात को चमकता है"]
    result = compute_bleu(refs, hyps, smooth=False)

    # Disjoint vocabulary has zero 2-gram, 3-gram, 4-gram matches
    assert result["bleu"] == 0.0


def test_bleu_brevity_penalty():
    refs = ["एक दो तीन चार पाँच छह सात आठ"]
    hyps = ["एक दो तीन"]
    result = compute_bleu(refs, hyps)

    # Hypothesis is significantly shorter than reference -> brevity penalty < 1.0
    assert result["brevity_penalty"] < 1.0
    assert result["hyp_len"] == 3
    assert result["ref_len"] == 8


def test_chrf_identical():
    refs = ["नमस्ते भारत"]
    hyps = ["नमस्ते भारत"]
    result = compute_chrf(refs, hyps)

    assert result["chrf"] == 1.0
    assert result["avg_precision"] == 1.0
    assert result["avg_recall"] == 1.0


def test_chrf_partial_overlap():
    refs = ["हमारा देश भारत है"]
    hyps = ["हमारा देश महान है"]
    result = compute_chrf(refs, hyps)

    assert 0.0 < result["chrf"] < 1.0
    assert result["avg_precision"] > 0.0
    assert result["avg_recall"] > 0.0


def test_token_accuracy():
    refs = [[10, 20, 30, 40, 0], [100, 200, 0, 0, 0]]
    preds = [[10, 20, 99, 40, 0], [100, 200, 0, 0, 0]]

    # Sample 1: tokens 10, 20, 40 match (3/4). Sample 2: 100, 200 match (2/2).
    # Total non-pad tokens = 6, matches = 5 -> accuracy = 5/6 = 0.833333
    result = token_accuracy(refs, preds, ignore_id=0)

    assert result["total"] == 6
    assert result["correct"] == 5
    assert abs(result["accuracy"] - 5.0 / 6.0) < 1e-4


def test_token_accuracy_all_match():
    refs = [[1, 2, 3]]
    preds = [[1, 2, 3]]
    result = token_accuracy(refs, preds, ignore_id=0)

    assert result["accuracy"] == 1.0
    assert result["correct"] == 3
    assert result["total"] == 3

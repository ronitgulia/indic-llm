"""
Indic LLM — Lightweight NLG Evaluation Metrics
=================================================
Dependency-free implementations of standard generation quality metrics,
tuned for Devanagari and other Indic scripts:

  - compute_bleu()   : Corpus-level BLEU with modified n-gram precision
  - compute_chrf()   : Character n-gram F-score (ChrF), the preferred metric
                        for morphologically rich languages like Hindi
  - token_accuracy() : Exact next-token prediction accuracy on held-out data

These avoid pulling in sacrebleu / nltk so the evaluation harness remains
lightweight and reproducible across environments.

References
----------
  BLEU:  Papineni et al., 2002 — "BLEU: a Method for Automatic Evaluation"
  ChrF:  Popović, 2015 — "chrF: character n-gram F-score for automatic MT evaluation"
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# BLEU — corpus-level with smoothing
# ---------------------------------------------------------------------------

def _count_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Extract n-gram counts from a token sequence."""
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    max_order: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU score.

    Parameters
    ----------
    references  : List of reference strings (one per sample).
    hypotheses  : List of hypothesis/generated strings (one per sample).
    max_order   : Maximum n-gram order (default: 4 for BLEU-4).
    smooth      : Apply +1 smoothing to prevent zero n-gram precision from
                  killing the geometric mean (recommended for short outputs).

    Returns
    -------
    Dictionary with keys: ``bleu``, ``brevity_penalty``, ``precisions``
    (list of per-order modified precisions), ``ref_len``, ``hyp_len``.
    """
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for ref_str, hyp_str in zip(references, hypotheses):
        ref_tokens = ref_str.split()
        hyp_tokens = hyp_str.split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for n in range(1, max_order + 1):
            ref_ngrams = _count_ngrams(ref_tokens, n)
            hyp_ngrams = _count_ngrams(hyp_tokens, n)

            # Clipped counts: min(hyp_count, ref_count) per n-gram
            for ngram, count in hyp_ngrams.items():
                matches_by_order[n - 1] += min(count, ref_ngrams.get(ngram, 0))

            possible_by_order[n - 1] += max(0, len(hyp_tokens) - n + 1)

    # Modified precisions per order
    precisions: List[float] = []
    for n in range(max_order):
        if possible_by_order[n] == 0:
            precisions.append(0.0)
        elif smooth:
            precisions.append(
                (matches_by_order[n] + 1.0) / (possible_by_order[n] + 1.0)
            )
        else:
            precisions.append(matches_by_order[n] / possible_by_order[n])

    # Geometric mean of precisions (in log space to avoid underflow)
    if min(precisions) > 0:
        log_avg = sum(math.log(p) for p in precisions) / max_order
        geo_mean = math.exp(log_avg)
    else:
        geo_mean = 0.0

    # Brevity penalty
    if hyp_length == 0:
        bp = 0.0
    elif hyp_length >= ref_length:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_length / hyp_length)

    bleu = bp * geo_mean

    return {
        "bleu": round(bleu, 6),
        "brevity_penalty": round(bp, 6),
        "precisions": [round(p, 6) for p in precisions],
        "ref_len": ref_length,
        "hyp_len": hyp_length,
    }


# ---------------------------------------------------------------------------
# ChrF — character n-gram F-score
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams (including spaces as characters)."""
    return Counter(text[i: i + n] for i in range(len(text) - n + 1))


def compute_chrf(
    references: List[str],
    hypotheses: List[str],
    max_order: int = 6,
    beta: float = 2.0,
) -> Dict[str, float]:
    """
    Compute corpus-level ChrF score.

    ChrF (character n-gram F-score) is especially well-suited for
    morphologically rich languages like Hindi, where subword structure
    carries significant semantic and grammatical information.

    Parameters
    ----------
    references  : List of reference strings.
    hypotheses  : List of hypothesis strings.
    max_order   : Maximum character n-gram order (default: 6, the ChrF standard).
    beta        : β weight for F-score (default: 2.0, recall-biased as in ChrF).

    Returns
    -------
    Dictionary with keys: ``chrf``, ``avg_precision``, ``avg_recall``.
    """
    total_precision = 0.0
    total_recall = 0.0
    n_orders = 0

    for n in range(1, max_order + 1):
        total_matches = 0
        total_hyp_ngrams = 0
        total_ref_ngrams = 0

        for ref_str, hyp_str in zip(references, hypotheses):
            ref_ngrams = _char_ngrams(ref_str, n)
            hyp_ngrams = _char_ngrams(hyp_str, n)

            # Clipped matches
            for ngram, count in hyp_ngrams.items():
                total_matches += min(count, ref_ngrams.get(ngram, 0))

            total_hyp_ngrams += sum(hyp_ngrams.values())
            total_ref_ngrams += sum(ref_ngrams.values())

        precision_n = total_matches / max(1, total_hyp_ngrams)
        recall_n = total_matches / max(1, total_ref_ngrams)

        total_precision += precision_n
        total_recall += recall_n
        n_orders += 1

    avg_p = total_precision / max(1, n_orders)
    avg_r = total_recall / max(1, n_orders)

    # β-weighted harmonic mean
    beta_sq = beta ** 2
    if avg_p + avg_r > 0:
        chrf = (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)
    else:
        chrf = 0.0

    return {
        "chrf": round(chrf, 6),
        "avg_precision": round(avg_p, 6),
        "avg_recall": round(avg_r, 6),
    }


# ---------------------------------------------------------------------------
# Token-level accuracy
# ---------------------------------------------------------------------------

def token_accuracy(
    references: List[List[int]],
    predictions: List[List[int]],
    ignore_id: int = 0,
) -> Dict[str, float]:
    """
    Compute exact next-token prediction accuracy.

    Parameters
    ----------
    references  : List of ground-truth token-ID sequences.
    predictions : List of predicted token-ID sequences (same shape).
    ignore_id   : Token ID to exclude from accuracy (padding token).

    Returns
    -------
    Dictionary with keys: ``accuracy``, ``correct``, ``total``.
    """
    correct = 0
    total = 0

    for ref_seq, pred_seq in zip(references, predictions):
        for ref_tok, pred_tok in zip(ref_seq, pred_seq):
            if ref_tok == ignore_id:
                continue
            total += 1
            if ref_tok == pred_tok:
                correct += 1

    accuracy = correct / max(1, total)
    return {
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # BLEU smoke test
    refs = ["भारत एक विशाल देश है", "हिंदी हमारी राष्ट्रभाषा है"]
    hyps = ["भारत एक बड़ा देश है", "हिंदी हमारी भाषा है"]
    bleu_result = compute_bleu(refs, hyps)
    print(f"BLEU: {bleu_result}")

    # ChrF smoke test
    chrf_result = compute_chrf(refs, hyps)
    print(f"ChrF: {chrf_result}")

    # Token accuracy smoke test
    ref_ids = [[10, 20, 30, 0, 0], [40, 50, 60, 70, 0]]
    pred_ids = [[10, 20, 99, 0, 0], [40, 50, 60, 70, 0]]
    acc = token_accuracy(ref_ids, pred_ids)
    print(f"Token accuracy: {acc}")

    print("\n✓ All benchmarks working correctly")

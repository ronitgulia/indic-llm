"""Extended Indic Evaluation Suite: BLEU, ChrF++, Reasoning & Transliteration Benchmarks.

Provides standardized multi-metric evaluation for Indic LLMs across translation,
step-by-step reasoning, cross-lingual transfer, and script transliteration fidelity.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MetricResult:
    """Encapsulates evaluation score and per-sample diagnostics."""
    metric_name: str
    score: float
    num_samples: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize_indic(text: str) -> list[str]:
    """Tokenize Indic text preserving punctuation and Unicode characters."""
    return [w for w in re.findall(r"[\w\u0900-\u0D7F]+|[^\w\s]", text) if w.strip()]


def compute_bleu(
    hypotheses: list[str],
    references: list[list[str]],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Calculate multi-reference sentence or corpus BLEU score (0.0 to 100.0)."""
    if not hypotheses or not references or len(hypotheses) != len(references):
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len_total = 0
    ref_len_total = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = tokenize_indic(hyp)
        ref_tokens_list = [tokenize_indic(r) for r in refs]

        hyp_len = len(hyp_tokens)
        hyp_len_total += hyp_len
        # Closest reference length for brevity penalty
        closest_ref_len = min(
            (len(r) for r in ref_tokens_list),
            key=lambda ref_l: (abs(ref_l - hyp_len), ref_l),
            default=0,
        )
        ref_len_total += closest_ref_len

        for n in range(1, max_n + 1):
            hyp_ngrams = collections.Counter(
                tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)
            )
            max_ref_ngrams: dict[tuple[str, ...], int] = {}
            for r_toks in ref_tokens_list:
                r_ngrams = collections.Counter(
                    tuple(r_toks[i:i + n]) for i in range(len(r_toks) - n + 1)
                )
                for ng, count in r_ngrams.items():
                    max_ref_ngrams[ng] = max(max_ref_ngrams.get(ng, 0), count)

            for ng, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, max_ref_ngrams.get(ng, 0))
            total_counts[n - 1] += max(0, len(hyp_tokens) - n + 1)

    # Compute precisions
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            p = 0.0
        else:
            p = (clipped_counts[n] + (1.0 if smooth else 0.0)) / (
                total_counts[n] + (1.0 if smooth else 0.0)
            )
        precisions.append(p)

    if min(precisions) <= 0.0 and not smooth:
        return 0.0

    # Log average precision
    log_prec_sum = sum(math.log(p) for p in precisions) / max_n
    geo_mean = math.exp(log_prec_sum)

    # Brevity penalty
    if hyp_len_total == 0:
        bp = 0.0
    elif hyp_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len_total / hyp_len_total))

    return round(bp * geo_mean * 100.0, 2)


def compute_chrf(
    hypotheses: list[str],
    references: list[list[str]],
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """Calculate corpus ChrF++ score across character and word n-grams (0.0 to 100.0)."""
    if not hypotheses or not references:
        return 0.0

    total_char_f = 0.0
    for hyp, refs in zip(hypotheses, references):
        hyp_chars = [c for c in hyp if not c.isspace()]
        ref_chars_list = [[c for c in r if not c.isspace()] for r in refs]

        sent_precisions = []
        sent_recalls = []

        for n in range(1, char_order + 1):
            hyp_ngrams = collections.Counter(
                "".join(hyp_chars[i:i + n]) for i in range(len(hyp_chars) - n + 1)
            )
            # Match against primary reference
            ref_ngrams = collections.Counter(
                "".join(ref_chars_list[0][i:i + n]) for i in range(len(ref_chars_list[0]) - n + 1)
            )
            matched = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
            total_hyp = max(0, len(hyp_chars) - n + 1)
            total_ref = max(0, len(ref_chars_list[0]) - n + 1)

            p = matched / total_hyp if total_hyp > 0 else 0.0
            r = matched / total_ref if total_ref > 0 else 0.0
            sent_precisions.append(p)
            sent_recalls.append(r)

        avg_p = sum(sent_precisions) / char_order if sent_precisions else 0.0
        avg_r = sum(sent_recalls) / char_order if sent_recalls else 0.0

        if (avg_p + avg_r) > 0:
            f = ((1 + beta ** 2) * avg_p * avg_r) / ((beta ** 2 * avg_p) + avg_r)
        else:
            f = 0.0
        total_char_f += f

    return round((total_char_f / len(hypotheses)) * 100.0, 2)


class IndicReasoningBenchmark:
    """Evaluates multi-step Indic mathematical and logical reasoning tasks."""

    def __init__(self, tasks_file: str | Path | None = None):
        self.tasks_file = Path(tasks_file) if tasks_file else None

    def evaluate_predictions(
        self,
        predictions: list[dict[str, Any]],
    ) -> MetricResult:
        """Evaluate accuracy of step-by-step reasoning outputs."""
        if not predictions:
            return MetricResult("indic_reasoning_acc", 0.0, 0, {})

        correct = 0
        total = len(predictions)
        for item in predictions:
            pred_ans = str(item.get("predicted_answer", "")).strip().lower()
            gold_ans = str(item.get("ground_truth", "")).strip().lower()
            if pred_ans == gold_ans or gold_ans in pred_ans:
                correct += 1

        acc = round((correct / total) * 100.0, 2)
        return MetricResult(
            metric_name="indic_reasoning_acc",
            score=acc,
            num_samples=total,
            metadata={"correct": correct, "total": total},
        )


class TransliterationAccuracyMetric:
    """Measures character-level and token-level transliteration fidelity."""

    @staticmethod
    def compute_accuracy(hypotheses: list[str], references: list[str]) -> float:
        if not hypotheses or not references or len(hypotheses) != len(references):
            return 0.0

        matches = 0
        for h, r in zip(hypotheses, references):
            if h.strip() == r.strip():
                matches += 1
        return round((matches / len(hypotheses)) * 100.0, 2)


class CrossLingualTransferScore:
    """Measures performance parity between English source queries and Indic translated queries."""

    @staticmethod
    def compute_parity(source_scores: list[float], target_scores: list[float]) -> float:
        if not source_scores or not target_scores or len(source_scores) != len(target_scores):
            return 0.0
        avg_src = sum(source_scores) / len(source_scores)
        avg_tgt = sum(target_scores) / len(target_scores)
        if avg_src == 0.0:
            return 0.0
        return round(min(1.0, avg_tgt / avg_src) * 100.0, 2)

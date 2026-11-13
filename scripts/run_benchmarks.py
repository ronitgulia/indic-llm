"""Automated regression runner for Indic LLM benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.indic_benchmarks import (  # noqa: E402
    IndicReasoningBenchmark,
    TransliterationAccuracyMetric,
    compute_bleu,
    compute_chrf,
)


def run_all():
    print("Running automated benchmark regression checks...")
    # 1. Translation benchmarks
    bleu = compute_bleu(["भारत सुंदर देश है।"], [["भारत सुंदर देश है।"]])
    chrf = compute_chrf(["भारत सुंदर देश है।"], [["भारत सुंदर देश है।"]])
    print(f"BLEU: {bleu}, ChrF++: {chrf}")
    assert bleu >= 95.0, "BLEU regression detected"
    assert chrf >= 95.0, "ChrF++ regression detected"

    # 2. Reasoning benchmark
    bench = IndicReasoningBenchmark()
    sample_preds = [
        {"predicted_answer": "10 फल बचे", "ground_truth": "10"},
        {"predicted_answer": "4", "ground_truth": "4"},
    ]
    res = bench.evaluate_predictions(sample_preds)
    print(f"Reasoning accuracy: {res.score}%")
    assert res.score == 100.0, "Reasoning regression detected"

    # 3. Transliteration accuracy
    trans_acc = TransliterationAccuracyMetric.compute_accuracy(["नमस्ते"], ["नमस्ते"])
    print(f"Transliteration accuracy: {trans_acc}%")
    assert trans_acc == 100.0, "Transliteration regression detected"

    print("All benchmark assertions passed cleanly!")


if __name__ == "__main__":
    run_all()

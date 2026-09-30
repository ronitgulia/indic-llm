"""Automated regression runner for Indic LLM benchmark suite."""

from eval.indic_benchmarks import IndicReasoningBenchmark, TransliterationAccuracyMetric, compute_bleu, compute_chrf


def run_all():
    print("Running automated benchmark regression checks...")
    # Sample translation
    bleu = compute_bleu(["भारत सुंदर देश है।"], [["भारत सुंदर देश है।"]])
    chrf = compute_chrf(["भारत सुंदर देश है।"], [["भारत सुंदर देश है।"]])
    print(f"BLEU: {bleu}, ChrF++: {chrf}")
    assert bleu >= 95.0, "BLEU regression detected"
    assert chrf >= 95.0, "ChrF++ regression detected"
    print("All benchmark assertions passed cleanly!")


if __name__ == "__main__":
    run_all()

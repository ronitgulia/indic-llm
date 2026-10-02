"""End-to-End integration test covering full Indic LLM pipeline."""

import pytest
from src.guardrails import IndicSafetyFilter
from src.transliterate import IndicTransliterator
from eval.indic_benchmarks import compute_bleu, compute_chrf
from inference.web_ui import IndicRequestHandler


def test_full_indic_pipeline():
    # 1. Transliteration
    trans = IndicTransliterator()
    devanagari_text = trans.transliterate("bharat", "Latin", "Devanagari")
    assert len(devanagari_text) > 0

    # 2. Safety Filtering
    guard = IndicSafetyFilter()
    verdict = guard.check_text("भारत एक सुंदर और समृद्ध देश है।")
    assert verdict.is_safe

    # 3. Evaluation
    bleu = compute_bleu(["भारत एक सुंदर देश है।"], [["भारत एक सुंदर देश है।"]])
    assert bleu >= 99.0
    chrf = compute_chrf(["भारत एक सुंदर देश है।"], [["भारत एक सुंदर देश है।"]])
    assert chrf >= 99.0

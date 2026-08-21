"""
Unit tests for Indic LLM Safety & Guardrails:
  - PII masking (Aadhaar, PAN, Phone, Email)
  - Devanagari profanity and toxicity detection
  - Prompt injection pattern matching
  - Dual-sided IndicGuardrails input/output validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from guardrails import (  # noqa: E402
    GuardrailResult,
    IndicGuardrails,
    PIIMasker,
    PIIType,
    PromptInjectionDetector,
    ToxicityClassifier,
)


def test_pii_masker_entities():
    masker = PIIMasker()
    text = "मेरा पैन कार्ड ABCDE1234F और फोन नंबर 9876543210 है। ईमेल test@example.com है।"
    masked, matches = masker.mask(text)

    assert "[पैन कार्ड]" in masked
    assert "[फोन नंबर]" in masked
    assert "[ईमेल]" in masked
    assert len(matches) == 3

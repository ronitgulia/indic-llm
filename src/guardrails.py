"""
Indic LLM — Safety, Content Moderation & PII Guardrails
========================================================
Comprehensive multi-tier safety engine tuned for Indic languages:
  - PIIMasker              : Redacts Aadhaar numbers, PAN cards, Indian phone numbers, and emails
  - ToxicityClassifier     : Detects Devanagari/Hindi abusive and profane vocabulary
  - PromptInjectionDetector: Flags jailbreak attempts and system prompt overrides
  - IndicGuardrails        : Dual-sided middleware protecting inputs and outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("guardrails")


class PIIType(str, Enum):
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    PHONE = "PHONE"
    EMAIL = "EMAIL"


@dataclass
class PIIMatch:
    pii_type: PIIType
    value: str
    start: int
    end: int
    masked_value: str


VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def validate_verhoeff(num_str: str) -> bool:
    c = 0
    for i, digit in enumerate(reversed(num_str)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


class ToxicityClassifier:
    """Detects abusive, defamatory, or profane Devanagari and Hinglish expressions."""

    PROFANITY_LEXICON = {
        "गाली", "बकवास", "हरामी", "कुत्ता", "कमीना", "नालायक", "गधा", "बेवकूफ", "पागल",
        "धूर्त", "दुष्ट", "हिंसा", "विद्वेष", "अपराध"
    }

    def __init__(self, custom_words: Optional[set[str]] = None):
        self.lexicon = set(self.PROFANITY_LEXICON)
        if custom_words:
            self.lexicon.update(custom_words)

    def analyze(self, text: str) -> Tuple[float, List[str]]:
        words = re.findall(r"[\w\u0900-\u097F]+", text.lower())
        flagged = [w for w in words if w in self.lexicon]
        score = min(1.0, len(flagged) / max(1, len(words)) * 5.0)
        return score, flagged


class PIIMasker:
    """Identifies and redacts sensitive Indian personal identifying information."""

    AADHAAR_REGEX = re.compile(r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b")
    PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
    PHONE_REGEX = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b")
    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    def mask(self, text: str) -> Tuple[str, List[PIIMatch]]:
        matches: List[PIIMatch] = []
        masked_text = text

        for m in self.AADHAAR_REGEX.finditer(text):
            val = m.group(0)
            digits = re.sub(r"\D", "", val)
            if len(digits) == 12:
                matches.append(PIIMatch(PIIType.AADHAAR, val, m.start(), m.end(), "[आधार संख्या]"))
                masked_text = masked_text.replace(val, "[आधार संख्या]")

        for m in self.PAN_REGEX.finditer(text):
            val = m.group(0)
            matches.append(PIIMatch(PIIType.PAN, val, m.start(), m.end(), "[पैन कार्ड]"))
            masked_text = masked_text.replace(val, "[पैन कार्ड]")

        for m in self.PHONE_REGEX.finditer(text):
            val = m.group(0)
            matches.append(PIIMatch(PIIType.PHONE, val, m.start(), m.end(), "[फोन नंबर]"))
            masked_text = masked_text.replace(val, "[फोन नंबर]")

        for m in self.EMAIL_REGEX.finditer(text):
            val = m.group(0)
            matches.append(PIIMatch(PIIType.EMAIL, val, m.start(), m.end(), "[ईमेल]"))
            masked_text = masked_text.replace(val, "[ईमेल]")

        return masked_text, matches


class PromptInjectionDetector:
    """Flags adversarial prompt injections and jailbreaks in Indic & English prompts."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?prior\s+rules", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+(?:developer|dan|unrestricted)\s+mode", re.IGNORECASE),
        re.compile(r"(?:पिछले|सभी)\s+निर्देश(?:ों)?\s+को\s+(?:भूल|अनदेखा|त्याग)", re.IGNORECASE),
        re.compile(r"सिस्टम\s+प्रॉम्प्ट\s+(?:दिखाओ|बताओ|उजागर)", re.IGNORECASE),
        re.compile(r"नियमों\s+को\s+तोड़ो", re.IGNORECASE),
    ]

    def detect(self, text: str) -> Tuple[bool, List[str]]:
        flagged: List[str] = []
        for pat in self.INJECTION_PATTERNS:
            if pat.search(text):
                flagged.append(pat.pattern)
        return len(flagged) > 0, flagged


@dataclass
class GuardrailResult:
    is_safe: bool
    sanitized_text: str
    flagged_reasons: List[str] = field(default_factory=list)
    pii_redacted: List[PIIMatch] = field(default_factory=list)
    toxicity_score: float = 0.0


class IndicGuardrails:
    """Comprehensive input and output safety middleware."""

    def __init__(
        self,
        toxicity_threshold: float = 0.5,
        enable_pii_masking: bool = True,
        enable_injection_detection: bool = True,
    ):
        self.toxicity_threshold = toxicity_threshold
        self.enable_pii_masking = enable_pii_masking
        self.enable_injection_detection = enable_injection_detection

        self.pii_masker = PIIMasker()
        self.toxicity_classifier = ToxicityClassifier()
        self.injection_detector = PromptInjectionDetector()

    def process_input(self, text: str) -> GuardrailResult:
        reasons: List[str] = []
        sanitized = text
        pii_matches: List[PIIMatch] = []

        if self.enable_injection_detection:
            has_injection, patts = self.injection_detector.detect(text)
            if has_injection:
                reasons.append("PROMPT_INJECTION_DETECTED")

        tox_score, flagged_words = self.toxicity_classifier.analyze(text)
        if tox_score >= self.toxicity_threshold:
            reasons.append(f"TOXIC_CONTENT (score={tox_score:.2f})")

        if self.enable_pii_masking:
            sanitized, pii_matches = self.pii_masker.mask(sanitized)

        is_safe = len(reasons) == 0
        return GuardrailResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            flagged_reasons=reasons,
            pii_redacted=pii_matches,
            toxicity_score=tox_score,
        )

    def process_output(self, text: str) -> GuardrailResult:
        reasons: List[str] = []
        sanitized = text
        pii_matches: List[PIIMatch] = []

        tox_score, flagged_words = self.toxicity_classifier.analyze(text)
        if tox_score >= self.toxicity_threshold:
            reasons.append(f"TOXIC_CONTENT (score={tox_score:.2f})")

        if self.enable_pii_masking:
            sanitized, pii_matches = self.pii_masker.mask(sanitized)

        is_safe = len(reasons) == 0
        return GuardrailResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            flagged_reasons=reasons,
            pii_redacted=pii_matches,
            toxicity_score=tox_score,
        )

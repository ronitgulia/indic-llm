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

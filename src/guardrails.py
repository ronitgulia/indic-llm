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

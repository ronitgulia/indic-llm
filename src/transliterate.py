"""
Indic LLM — Multi-Script & Phonetic Transliteration Engine
==========================================================
Enables cross-lingual processing across 8 major Indic scripts and Romanized (Hinglish) text:
  - ScriptType      : Enum representing Indic scripts and Unicode blocks
  - detect_script   : Identifies predominant script using Unicode range heuristics
  - roman_to_indic  : Phonetic mapping for Romanized (Hinglish/IAST) text to Devanagari
  - cross_indic     : Converts between Devanagari and regional Indic scripts
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transliterate")


class ScriptType(str, Enum):
    DEVANAGARI = "Devanagari"
    BENGALI = "Bengali"
    GURMUKHI = "Gurmukhi"
    GUJARATI = "Gujarati"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    KANNADA = "Kannada"
    MALAYALAM = "Malayalam"
    LATIN = "Latin"
    UNKNOWN = "Unknown"


UNICODE_RANGES: Dict[ScriptType, Tuple[int, int]] = {
    ScriptType.DEVANAGARI: (0x0900, 0x097F),
    ScriptType.BENGALI: (0x0980, 0x09FF),
    ScriptType.GURMUKHI: (0x0A00, 0x0A7F),
    ScriptType.GUJARATI: (0x0A80, 0x0AFF),
    ScriptType.TAMIL: (0x0B80, 0x0BFF),
    ScriptType.TELUGU: (0x0C00, 0x0C7F),
    ScriptType.KANNADA: (0x0C80, 0x0CFF),
    ScriptType.MALAYALAM: (0x0D00, 0x0D7F),
    ScriptType.LATIN: (0x0041, 0x007A),
}


def detect_script(text: str) -> ScriptType:
    counts: Dict[ScriptType, int] = {st: 0 for st in UNICODE_RANGES}
    for ch in text:
        cp = ord(ch)
        for st, (start, end) in UNICODE_RANGES.items():
            if start <= cp <= end:
                counts[st] += 1
                break

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_counts[0][1] > 0:
        return sorted_counts[0][0]
    return ScriptType.UNKNOWN


ROMAN_TO_DEV_MAP = {
    "aa": "ा", "ee": "ी", "oo": "ू", "ai": "ै", "au": "ौ",
    "kh": "ख", "gh": "घ", "ch": "च", "chh": "छ", "jh": "झ",
    "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ", "sh": "श",
    "k": "क", "g": "ग", "j": "ज", "t": "त", "d": "द",
    "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य",
    "r": "र", "l": "ल", "v": "व", "w": "व", "s": "स",
    "h": "ह", "a": "", "i": "ि", "u": "ु", "e": "े", "o": "ो",
}

ROMAN_INITIAL_VOWELS = {
    "aa": "आ", "ee": "ई", "oo": "ऊ", "ai": "ऐ", "au": "औ",
    "a": "अ", "i": "इ", "u": "उ", "e": "ए", "o": "ओ",
}


def roman_to_devanagari(text: str) -> str:
    words = text.split()
    converted_words = []

    for word in words:
        clean_w = word.lower()
        res = ""
        idx = 0
        w_len = len(clean_w)
        is_first = True

        while idx < w_len:
            matched = False
            for l in (3, 2, 1):
                sub = clean_w[idx : idx + l]
                if is_first and sub in ROMAN_INITIAL_VOWELS:
                    res += ROMAN_INITIAL_VOWELS[sub]
                    idx += l
                    matched = True
                    is_first = False
                    break
                elif sub in ROMAN_TO_DEV_MAP:
                    res += ROMAN_TO_DEV_MAP[sub]
                    idx += l
                    matched = True
                    is_first = False
                    break
            if not matched:
                res += clean_w[idx]
                idx += 1
                is_first = False
        converted_words.append(res)

    return " ".join(converted_words)

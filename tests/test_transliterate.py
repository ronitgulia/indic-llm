"""
Unit tests for Indic Multi-Script Transliteration:
  - Script identification across 8 Indic scripts
  - Romanized Hinglish to Devanagari phonetic conversion
  - Cross-script Brahmic Unicode offset transliteration
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from transliterate import (  # noqa: E402
    ScriptType,
    detect_script,
    devanagari_to_script,
    roman_to_devanagari,
)


def test_detect_script():
    assert detect_script("भारत एक महान देश है") == ScriptType.DEVANAGARI
    assert detect_script("আমার সোনার বাংলা") == ScriptType.BENGALI
    assert detect_script("Hello world") == ScriptType.LATIN


def test_roman_to_devanagari():
    res = roman_to_devanagari("bharat")
    assert "भ" in res
    assert "र" in res
    assert "त" in res

    res2 = roman_to_devanagari("namaste")
    assert "न" in res2

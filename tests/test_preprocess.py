"""
Unit tests for Indic text preprocessing pipeline:
  - Unicode NFC normalisation
  - Noise & HTML stripping
  - Devanagari script ratio computation
  - Heuristic quality filtering
  - Deduplication
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (
    IndicTextCleaner,
    QualityConfig,
    QualityFilter,
    _extract_instruct,
    indic_char_count,
    normalize_unicode,
    script_ratio,
    strip_noise,
)


def test_normalize_unicode():
    # Composed vs decomposed forms (NFD -> NFC)
    raw = "हिंदी"
    decomposed = unicodedata.normalize("NFD", raw)
    normalized = normalize_unicode(decomposed)

    assert unicodedata.is_normalized("NFC", normalized)
    assert normalized == raw


def test_strip_noise():
    dirty = "Check this link: https://example.com/info <p>नमस्ते</p>   भारत!!!"
    cleaned = strip_noise(dirty)

    assert "https://" not in cleaned
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "नमस्ते" in cleaned
    assert "भारत" in cleaned
    # Multiple spaces collapsed
    assert "   " not in cleaned


def test_script_ratio():
    hindi_only = "नमस्ते भारत"
    h_ratio, l_ratio = script_ratio(hindi_only)
    assert h_ratio == 1.0
    assert l_ratio == 0.0

    english_only = "Hello World"
    h_ratio, l_ratio = script_ratio(english_only)
    assert h_ratio == 0.0
    assert l_ratio == 1.0

    code_mix = "Hello भारत"
    h_ratio, l_ratio = script_ratio(code_mix)
    assert 0.0 < h_ratio < 1.0
    assert 0.0 < l_ratio < 1.0


def test_quality_filter():
    qf = QualityFilter(QualityConfig(min_chars=10, min_hindi_ratio=0.4))

    # Valid Hindi sentence
    assert qf.is_valid("यह एक बहुत अच्छी और ज्ञानवर्धक किताब है।")

    # Too short
    assert not qf.is_valid("छोटा")

    # Pure English fails min_hindi_ratio
    assert not qf.is_valid("This is a completely english document without hindi text.")

    # Excessive character repetition
    assert not qf.is_valid("नमस्ते अअअअअअअअअअअअअ भारत")


def test_cleaner_deduplication():
    cleaner = IndicTextCleaner(dedup=True)
    sample = "भारत एक प्राचीन और महान संस्कृति वाला देश है।"

    res1 = cleaner.clean(sample)
    assert res1 is not None

    # Duplicate should be filtered out
    res2 = cleaner.clean(sample)
    assert res2 is None
    assert cleaner.n_duplicate == 1


def test_multilingual_indic_preservation():
    # Bengali, Tamil, Telugu must not be stripped as noise
    bengali = "গাছ পরিবেশের ভারসাম্য রক্ষা করে।"
    tamil = "தண்ணீர் ஒரு விலைமதிப்பற்ற இயற்கை வளம்."
    telugu = "వ్యాయామం యొక్క ప్రాముఖ్యత ఏమిటి?"

    clean_bn = strip_noise(bengali)
    clean_ta = strip_noise(tamil)
    clean_te = strip_noise(telugu)

    assert "গাছ" in clean_bn
    assert "தண்ணீர்" in clean_ta
    assert "వ్యాయామం" in clean_te

    assert indic_char_count(bengali) > 10
    assert indic_char_count(tamil) > 10
    assert indic_char_count(telugu) > 10


def test_extract_instruct_schemas():
    # 1. Direct instruction/response
    rec1 = {"instruction": "प्रश्न", "response": "उत्तर"}
    assert _extract_instruct(rec1) == "[INST] प्रश्न [/INST] उत्तर"

    # 2. Messages list
    rec2 = {
        "messages": [
            {"role": "user", "content": "क्या हाल है?"},
            {"role": "assistant", "content": "सब ठीक है।"},
        ]
    }
    assert _extract_instruct(rec2) == "[INST] क्या हाल है? [/INST] सब ठीक है।"

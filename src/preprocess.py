"""
Indic LLM — Text Preprocessing Pipeline
=========================================
Provides a composable, configurable cleaning pipeline for Indic text corpora:

  - Unicode NFC normalisation
  - URL / HTML / noise removal
  - Devanagari (and optionally multi-script) language filtering
  - Exact deduplication via SHA-256 hashing
  - Heuristic quality filtering (length, script ratio, repetition)
  - Per-dataset cleaner functions for the three supported schemas

Supported JSONL schemas
-----------------------
  Sangraha (monolingual):  {"text": "..."}
  IITB Parallel:           {"translation": {"en": "...", "hi": "..."}}
  Indic-Instruct:          {"messages": [{"role": "user"|"assistant", "content": "..."}]}

Usage (library):
    from src.preprocess import IndicTextCleaner, clean_corpus
    cleaner = IndicTextCleaner(min_hindi_chars=20, dedup=True)
    clean_corpus("data/raw/sangraha_hi.jsonl", "data/processed/sangraha_clean.jsonl",
                 schema="sangraha", cleaner=cleaner)

Usage (CLI):
    python src/preprocess.py --schema sangraha \
        --input data/raw/sangraha_hi.jsonl \
        --output data/processed/sangraha_clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unicode ranges of interest
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Allowed character classes in the output:
#   Devanagari block  \u0900-\u097F
#   Basic Latin       \u0020-\u007E   (printable ASCII)
#   Punctuation       various Unicode blocks kept via whitelist
_NOISE_CHARS_RE = re.compile(
    r"[^\u0900-\u097F"   # Devanagari
    r"\u0020-\u007E"     # Basic Latin (printable)
    r"\u00A0-\u00FF"     # Latin-1 supplement (accents etc.)
    r"\u2000-\u206F"     # General punctuation
    r"\n\r\t]"
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")   # 5+ consecutive identical chars


# ---------------------------------------------------------------------------
# Low-level text utilities
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalisation (critical for Devanagari composed forms)."""
    return unicodedata.normalize("NFC", text)


def strip_noise(text: str) -> str:
    """
    Remove URLs, HTML tags, noise characters, and collapse whitespace.
    Preserves Devanagari, Basic Latin, and common punctuation.
    """
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _NOISE_CHARS_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def devanagari_char_count(text: str) -> int:
    """Return the number of Devanagari code points in *text*."""
    return len(_DEVANAGARI_RE.findall(text))


def script_ratio(text: str) -> Tuple[float, float]:
    """
    Return (hindi_ratio, latin_ratio) as fractions of total non-space chars.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0, 0.0
    n = len(chars)
    h = sum(1 for c in chars if _DEVANAGARI_RE.match(c))
    l_ = sum(1 for c in chars if _LATIN_RE.match(c))
    return h / n, l_ / n


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


@dataclass
class QualityConfig:
    """Configurable thresholds for the quality filter."""

    min_chars: int = 20                  # minimum total characters
    max_chars: int = 50_000              # maximum total characters
    min_hindi_chars: int = 10            # minimum Devanagari characters
    min_hindi_ratio: float = 0.0         # minimum fraction of Devanagari chars (0 = disabled)
    allow_repeated_chars: bool = False   # reject lines with 5+ repeated identical chars
    max_latin_ratio: float = 0.9         # reject nearly-all-Latin text (not Indic)


class QualityFilter:
    """
    Applies heuristic quality checks to a cleaned text string.

    Parameters
    ----------
    config : QualityConfig
        Threshold configuration object.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        self.cfg = config or QualityConfig()

    def is_valid(self, text: str) -> bool:
        """Return True if *text* passes all quality checks."""
        if len(text) < self.cfg.min_chars:
            return False
        if len(text) > self.cfg.max_chars:
            return False
        if devanagari_char_count(text) < self.cfg.min_hindi_chars:
            return False

        h_ratio, l_ratio = script_ratio(text)
        if self.cfg.min_hindi_ratio > 0 and h_ratio < self.cfg.min_hindi_ratio:
            return False
        if l_ratio > self.cfg.max_latin_ratio:
            return False
        if not self.cfg.allow_repeated_chars and _REPEATED_CHAR_RE.search(text):
            return False

        return True


# ---------------------------------------------------------------------------
# IndicTextCleaner — main public class
# ---------------------------------------------------------------------------


class IndicTextCleaner:
    """
    Composable text-cleaning pipeline for Indic corpora.

    Steps (in order):
      1. Unicode NFC normalisation
      2. Noise stripping (URLs, HTML, forbidden chars)
      3. Quality filtering via QualityFilter
      4. Optional exact deduplication (SHA-256 of cleaned text)

    Parameters
    ----------
    quality_config : QualityConfig | None
        Thresholds for the quality filter. Uses safe defaults if None.
    dedup : bool
        If True, maintain an internal seen-set and skip duplicates.
    """

    def __init__(
        self,
        quality_config: Optional[QualityConfig] = None,
        dedup: bool = True,
    ):
        self._qf = QualityFilter(quality_config)
        self._dedup = dedup
        self._seen: Set[str] = set()

        # Counters
        self.n_seen: int = 0
        self.n_passed: int = 0
        self.n_too_short: int = 0
        self.n_low_quality: int = 0
        self.n_duplicate: int = 0

    def clean(self, text: str) -> Optional[str]:
        """
        Run the full cleaning pipeline on a single text string.

        Returns
        -------
        Cleaned string if it passes all filters, otherwise ``None``.
        """
        self.n_seen += 1

        text = normalize_unicode(text)
        text = strip_noise(text)

        if not self._qf.is_valid(text):
            self.n_low_quality += 1
            return None

        if self._dedup:
            digest = _sha256(text)
            if digest in self._seen:
                self.n_duplicate += 1
                return None
            self._seen.add(digest)

        self.n_passed += 1
        return text

    def reset_dedup(self) -> None:
        """Clear the deduplication cache (call between dataset shards if needed)."""
        self._seen.clear()

    def stats(self) -> Dict[str, int]:
        """Return a dictionary of processing statistics."""
        return {
            "seen": self.n_seen,
            "passed": self.n_passed,
            "low_quality": self.n_low_quality,
            "duplicate": self.n_duplicate,
        }

    def log_stats(self) -> None:
        """Emit statistics to the logger."""
        s = self.stats()
        retention = 100.0 * s["passed"] / max(1, s["seen"])
        log.info(
            "Stats — seen: %d | passed: %d (%.1f%%) | low_quality: %d | dup: %d",
            s["seen"], s["passed"], retention, s["low_quality"], s["duplicate"],
        )


# ---------------------------------------------------------------------------
# Per-schema extraction helpers
# ---------------------------------------------------------------------------


def _extract_sangraha(obj: dict) -> Optional[str]:
    """Extract text from Sangraha-style ``{"text": "..."}`` records."""
    return obj.get("text")


def _extract_parallel(obj: dict) -> Optional[str]:
    """
    Extract text from IITB parallel ``{"translation": {"hi": ..., "en": ...}}`` records.
    Returns the Hindi side (or Hindi + English concatenated if both present).
    """
    tr = obj.get("translation", {})
    hi = tr.get("hi", "").strip()
    en = tr.get("en", "").strip()
    if not hi:
        return None
    return f"{hi} {en}".strip() if en else hi


def _extract_instruct(obj: dict) -> Optional[str]:
    """
    Extract text from Indic-Instruct ``{"messages": [...]}`` records.
    Returns a formatted [INST] ... [/INST] string.
    """
    user_msg = ""
    assistant_msg = ""
    for msg in obj.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if role == "user":
            user_msg = content
        elif role == "assistant":
            assistant_msg = content

    if not user_msg or not assistant_msg:
        return None
    return f"[INST] {user_msg} [/INST] {assistant_msg}"


_SCHEMA_EXTRACTORS = {
    "sangraha": _extract_sangraha,
    "parallel": _extract_parallel,
    "instruct": _extract_instruct,
}


# ---------------------------------------------------------------------------
# Corpus cleaner
# ---------------------------------------------------------------------------


def clean_corpus(
    input_path: str,
    output_path: str,
    schema: str,
    cleaner: Optional[IndicTextCleaner] = None,
) -> Dict[str, int]:
    """
    Clean a single JSONL file and write a cleaned JSONL file.

    Parameters
    ----------
    input_path  : Source JSONL file path.
    output_path : Destination JSONL file path.
    schema      : One of ``"sangraha"``, ``"parallel"``, ``"instruct"``.
    cleaner     : Optional pre-configured IndicTextCleaner. A default one is
                  created if not provided.

    Returns
    -------
    dict with ``n_in`` and ``n_out`` counts.
    """
    if schema not in _SCHEMA_EXTRACTORS:
        raise ValueError(f"Unknown schema '{schema}'. Choose from {list(_SCHEMA_EXTRACTORS)}")

    extractor = _SCHEMA_EXTRACTORS[schema]
    cleaner = cleaner or IndicTextCleaner()

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Cleaning: %s (schema=%s)", input_path.name, schema)
    n_in = n_out = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for raw_line in fin:
            n_in += 1
            try:
                obj = json.loads(raw_line)
                raw_text = extractor(obj)
                if raw_text is None:
                    continue

                clean_text = cleaner.clean(raw_text)
                if clean_text is None:
                    continue

                # Write cleaned record preserving schema key
                if schema == "sangraha":
                    record = {"text": clean_text}
                elif schema == "parallel":
                    # Re-extract clean Hindi and English separately
                    tr = obj.get("translation", {})
                    hi_clean = cleaner.clean(normalize_unicode(tr.get("hi", "")))
                    en_text = tr.get("en", "").strip()
                    if hi_clean is None:
                        continue
                    record = {"hi": hi_clean, "en": en_text}
                    # Undo double-count from second clean() call
                    cleaner.n_seen -= 1
                    cleaner.n_passed -= 1
                else:  # instruct
                    parts = clean_text.split("[/INST]", 1)
                    instr = parts[0].replace("[INST]", "").strip()
                    out_text = parts[1].strip() if len(parts) > 1 else ""
                    record = {"instruction": instr, "output": out_text}

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_out += 1

            except Exception:
                continue

    log.info("  → %d / %d records retained (%.1f%%)", n_out, n_in, 100.0 * n_out / max(1, n_in))
    cleaner.log_stats()
    return {"n_in": n_in, "n_out": n_out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description="Indic LLM — text preprocessing pipeline",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--schema",
        required=True,
        choices=list(_SCHEMA_EXTRACTORS.keys()),
        help="Dataset schema (determines which fields to extract)",
    )
    parser.add_argument(
        "--min_hindi_chars",
        type=int,
        default=10,
        help="Minimum Devanagari characters per sample (default: 10)",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable exact deduplication",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    qcfg = QualityConfig(min_hindi_chars=args.min_hindi_chars)
    cleaner = IndicTextCleaner(quality_config=qcfg, dedup=not args.no_dedup)

    clean_corpus(
        input_path=args.input,
        output_path=args.output,
        schema=args.schema,
        cleaner=cleaner,
    )

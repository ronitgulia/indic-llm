"""
Indic LLM — Tokenizer Module
==============================
Wraps a SentencePiece BPE model with a clean Python API:
  - IndicTokenizer class for encode / decode / batch operations
  - Special-token registry (BOS, EOS, PAD, UNK, INST, system tags)
  - CLI with `train` and `test` sub-commands

Usage (library):
    from src.tokenizer import IndicTokenizer
    tok = IndicTokenizer("data/tokenizer/indic_spm.model")
    ids = tok.encode("नमस्ते दुनिया", add_bos=True, add_eos=True)

Usage (CLI):
    python src/tokenizer.py train --data_dir data/processed --output data/tokenizer/indic_spm
    python src/tokenizer.py test  --model  data/tokenizer/indic_spm.model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

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
# Special token constants
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "[INST]",
    "[/INST]",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

_DEFAULT_SPM_TRAIN_PARAMS: dict = dict(
    model_type="bpe",
    character_coverage=0.9999,          # full Devanagari + Latin coverage
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
    input_sentence_size=500_000,
    shuffle_input_sentence=True,
    num_threads=os.cpu_count() or 4,
)


# ---------------------------------------------------------------------------
# IndicTokenizer
# ---------------------------------------------------------------------------


class IndicTokenizer:
    """
    Thin wrapper around a trained SentencePiece model.

    Parameters
    ----------
    model_path : str | Path
        Path to the ``.model`` file produced by SentencePiece.

    Attributes
    ----------
    bos_id, eos_id, pad_id, unk_id : int
        IDs of the four reserved tokens.
    vocab_size : int
        Total vocabulary size including special tokens.
    """

    def __init__(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path}\n"
                "Run: python src/tokenizer.py train ..."
            )

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path))

        self.bos_id: int = self._sp.bos_id()
        self.eos_id: int = self._sp.eos_id()
        self.pad_id: int = self._sp.pad_id()
        self.unk_id: int = self._sp.unk_id()
        self.vocab_size: int = self._sp.get_piece_size()

        log.info(
            "IndicTokenizer loaded | vocab=%d | bos=%d eos=%d pad=%d unk=%d",
            self.vocab_size,
            self.bos_id,
            self.eos_id,
            self.pad_id,
            self.unk_id,
        )

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Parameters
        ----------
        text     : Input text (any Indic or Latin script).
        add_bos  : Prepend the BOS token ID.
        add_eos  : Append the EOS token ID.

        Returns
        -------
        List[int] of token IDs.
        """
        ids: List[int] = self._sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs back to a string.

        Parameters
        ----------
        ids           : Token ID sequence.
        skip_special  : If True, silently drop BOS/EOS/PAD tokens.

        Returns
        -------
        Decoded string.
        """
        if skip_special:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self._sp.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """Return the surface-form token strings (pieces)."""
        return self._sp.encode(text, out_type=str)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        pad_to_length: Optional[int] = None,
        truncate: bool = True,
    ) -> List[List[int]]:
        """
        Encode a batch of strings, with optional padding/truncation.

        Parameters
        ----------
        texts          : List of input strings.
        add_bos        : Prepend BOS to every sequence.
        add_eos        : Append EOS to every sequence.
        pad_to_length  : If set, pad shorter sequences with ``pad_id``.
        truncate       : Truncate sequences that exceed ``pad_to_length``.

        Returns
        -------
        List of token-ID lists (uniform length if ``pad_to_length`` is set).
        """
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        if pad_to_length is not None:
            result: List[List[int]] = []
            for seq in encoded:
                if truncate:
                    seq = seq[:pad_to_length]
                seq = seq + [self.pad_id] * max(0, pad_to_length - len(seq))
                result.append(seq)
            return result

        return encoded

    def decode_batch(self, batch: List[List[int]], skip_special: bool = True) -> List[str]:
        """Decode a list of token-ID sequences."""
        return [self.decode(ids, skip_special=skip_special) for ids in batch]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def piece_to_id(self, piece: str) -> int:
        """Look up a single piece string → token ID."""
        return self._sp.piece_to_id(piece)

    def id_to_piece(self, idx: int) -> str:
        """Look up a token ID → piece string."""
        return self._sp.id_to_piece(idx)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"IndicTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _collect_training_corpus(data_dir: Path, output_path: Path, max_lines: int = 500_000) -> Path:
    """
    Walk *data_dir* for ``*.jsonl`` files and extract raw text lines
    suitable for SentencePiece training.

    Supports three JSONL schemas:
      - ``{"text": "..."}``           (monolingual corpora such as Sangraha)
      - ``{"hi": "...", "en": "..."}``(parallel corpora)
      - ``{"instruction": "...", "output": "..."}`` (instruction-tuning)
    """
    log.info("Collecting training corpus from: %s", data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            log.info("  Reading: %s", jsonl_file.name)
            with jsonl_file.open("r", encoding="utf-8") as fin:
                for raw_line in fin:
                    if count >= max_lines:
                        break
                    try:
                        obj = json.loads(raw_line)
                        if "text" in obj:
                            text = obj["text"]
                        elif "hi" in obj:
                            text = obj["hi"]
                        elif "instruction" in obj:
                            text = f"{obj['instruction']} {obj.get('output', '')}"
                        else:
                            continue

                        text = text.strip()
                        if len(text) > 10:
                            fout.write(text + "\n")
                            count += 1
                    except Exception:
                        continue
            if count >= max_lines:
                break

    log.info("Corpus lines written: %d → %s", count, output_path)
    return output_path


def train_tokenizer(
    text_file: Union[str, Path],
    model_prefix: Union[str, Path] = "data/tokenizer/indic_spm",
    vocab_size: int = 32_000,
    extra_params: Optional[dict] = None,
) -> None:
    """
    Train a SentencePiece BPE tokenizer on a plain-text corpus.

    Parameters
    ----------
    text_file    : Path to the newline-separated corpus file.
    model_prefix : Output prefix — will produce ``{prefix}.model`` and ``{prefix}.vocab``.
    vocab_size   : Number of subword tokens (default: 32 000).
    extra_params : Additional SentencePiece trainer parameters to override defaults.
    """
    text_file = Path(text_file)
    if not text_file.exists():
        raise FileNotFoundError(f"Training corpus not found: {text_file}")

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    params = {**_DEFAULT_SPM_TRAIN_PARAMS}
    params.update(extra_params or {})
    params["user_defined_symbols"] = ",".join(SPECIAL_TOKENS)

    log.info("Training SentencePiece tokenizer | vocab=%d | input=%s", vocab_size, text_file)

    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        **params,
    )

    log.info("Tokenizer saved → %s.model", model_prefix)


# ---------------------------------------------------------------------------
# CLI test helper
# ---------------------------------------------------------------------------

_TEST_SENTENCES = [
    "नमस्ते, आप कैसे हैं?",
    "मैं एक AI भाषा मॉडल हूं।",
    "भारत एक विविधताओं से भरा देश है।",
    "Hello, this is a mixed-script sentence. नमस्ते!",
    "[INST] भारत की राजधानी क्या है? [/INST] भारत की राजधानी नई दिल्ली है।",
]


def _run_test(model_path: str) -> None:
    """Interactive smoke-test: encode → decode round-trip for a handful of sentences."""
    tok = IndicTokenizer(model_path)
    log.info("Running tokenizer round-trip test (%d sentences) …", len(_TEST_SENTENCES))

    ok = 0
    for sentence in _TEST_SENTENCES:
        ids = tok.encode(sentence, add_bos=True, add_eos=True)
        pieces = tok.tokenize(sentence)
        decoded = tok.decode(ids)

        match = decoded.strip() == sentence.strip()
        status = "✓" if match else "~"
        ok += int(match)

        print(f"\n  {status} Input   : {sentence}")
        print(f"    Pieces  : {pieces}")
        print(f"    IDs     : {ids[:12]}{'…' if len(ids) > 12 else ''}")
        print(f"    Decoded : {decoded}")

    print(f"\n  Vocab size : {tok.vocab_size:,}")
    print(f"  Round-trip : {ok}/{len(_TEST_SENTENCES)} exact matches")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenizer",
        description="Indic LLM — SentencePiece tokenizer trainer and tester",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train a new BPE tokenizer")
    train_p.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing *.jsonl processed data files",
    )
    train_p.add_argument(
        "--corpus_file",
        type=str,
        default=None,
        help="(optional) Pre-built corpus .txt file — skips corpus collection",
    )
    train_p.add_argument(
        "--output",
        type=str,
        default="data/tokenizer/indic_spm",
        help="Output model prefix (no extension)",
    )
    train_p.add_argument("--vocab_size", type=int, default=32_000)
    train_p.add_argument("--max_lines", type=int, default=500_000)

    # ── test ───────────────────────────────────────────────────────────────
    test_p = sub.add_parser("test", help="Run round-trip smoke test")
    test_p.add_argument(
        "--model",
        type=str,
        default="data/tokenizer/indic_spm.model",
        help="Path to trained .model file",
    )

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "train":
        corpus_file = args.corpus_file
        if corpus_file is None:
            corpus_file = str(
                _collect_training_corpus(
                    data_dir=Path(args.data_dir),
                    output_path=Path("data/tokenizer/corpus.txt"),
                    max_lines=args.max_lines,
                )
            )
        train_tokenizer(
            text_file=corpus_file,
            model_prefix=args.output,
            vocab_size=args.vocab_size,
        )

    elif args.command == "test":
        _run_test(args.model)
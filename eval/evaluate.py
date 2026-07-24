"""
Indic LLM — Evaluation Harness
================================
Provides:
  - PerplexityEvaluator  : Computes model perplexity on a held-out JSONL corpus
  - AccuracyEvaluator    : Token-level next-token prediction accuracy
  - GenerationQualityMetrics : Analyses generated text for repetition, script ratio,
                               BLEU, and ChrF (character F-score for Indic scripts)
  - run_eval() : Orchestrates all evaluators and writes a JSON report

Usage (CLI):
    python eval/evaluate.py \
        --checkpoint checkpoints/step_010000.pt \
        --data data/processed/sangraha_clean.jsonl \
        --tokenizer data/tokenizer/indic_spm.model \
        --prompts eval/human_eval/sample_prompts.jsonl \
        --output eval/results/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset

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
# Import model (handle running from different CWDs)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import IndicLLM, ModelConfig  # noqa: E402
from benchmarks import compute_bleu, compute_chrf, token_accuracy  # noqa: E402

# ---------------------------------------------------------------------------
# Utility: load a checkpoint
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> IndicLLM:
    """Load an IndicLLM from a checkpoint file."""
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    raw_cfg = ckpt.get("config", {})
    model_cfg = ModelConfig(
        vocab_size=raw_cfg.get("vocab_size", 32000),
        dim=raw_cfg.get("model_dim", 512),
        n_layers=raw_cfg.get("n_layers", 8),
        n_heads=raw_cfg.get("n_heads", 8),
        n_kv_heads=raw_cfg.get("n_kv_heads", 4),
        max_seq_len=raw_cfg.get("max_seq_len", 512),
        dropout=0.0,   # disable dropout at eval time
    )

    model = IndicLLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(
        "Model loaded — %.2fM params | step %d",
        model.num_parameters() / 1e6,
        ckpt.get("step", 0),
    )
    return model


# ---------------------------------------------------------------------------
# Perplexity Evaluator
# ---------------------------------------------------------------------------

class _PerplexityDataset(Dataset):
    """Tokenised fixed-length chunks of a JSONL corpus for perplexity evaluation."""

    def __init__(
        self,
        path: str,
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int,
        max_samples: int = 2000,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(self.samples) >= max_samples:
                    break
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("hi") or ""
                    if len(text) > 20:
                        self.samples.append(text)
                except Exception:
                    continue

        log.info("PerplexityDataset: %d samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.samples[idx], out_type=int)
        ids = ids[: self.max_seq_len + 1]
        ids += [0] * (self.max_seq_len + 1 - len(ids))
        return torch.tensor(ids, dtype=torch.long)


@dataclass
class PerplexityResult:
    avg_loss: float
    perplexity: float
    n_batches: int
    eval_time_s: float


class PerplexityEvaluator:
    """
    Compute token-level cross-entropy loss and perplexity on a held-out corpus.

    Parameters
    ----------
    model     : Trained IndicLLM in eval mode.
    tokenizer : SentencePiece model matching the training tokenizer.
    device    : Device to run evaluation on.
    """

    def __init__(
        self,
        model: IndicLLM,
        tokenizer: spm.SentencePieceProcessor,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        data_path: str,
        batch_size: int = 8,
        max_samples: int = 2000,
        max_seq_len: int = 512,
    ) -> PerplexityResult:
        """
        Evaluate perplexity on *data_path*.

        Returns
        -------
        PerplexityResult with avg_loss, perplexity, n_batches, eval_time_s.
        """
        dataset = _PerplexityDataset(data_path, self.tokenizer, max_seq_len, max_samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            tokens = batch.to(self.device)            # (B, T+1)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]

            _, loss = self.model(inputs, targets)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0

        log.info(
            "Perplexity — avg_loss: %.4f | ppl: %.2f | batches: %d | time: %.1fs",
            avg_loss, ppl, n_batches, elapsed,
        )
        return PerplexityResult(
            avg_loss=avg_loss,
            perplexity=ppl,
            n_batches=n_batches,
            eval_time_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Token-level Accuracy Evaluator
# ---------------------------------------------------------------------------


@dataclass
class AccuracyResult:
    accuracy: float
    correct: int
    total: int
    eval_time_s: float


class AccuracyEvaluator:
    """
    Compute token-level next-token prediction accuracy on held-out data.

    This measures the model's raw predictive accuracy — the fraction of
    non-padding tokens for which argmax(logits) exactly matches the target.
    Unlike perplexity, this gives an intuitive "percent correct" measure.

    Parameters
    ----------
    model     : Trained IndicLLM in eval mode.
    tokenizer : SentencePiece model matching the training tokenizer.
    device    : Device to run evaluation on.
    """

    def __init__(
        self,
        model: IndicLLM,
        tokenizer: spm.SentencePieceProcessor,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        data_path: str,
        batch_size: int = 8,
        max_samples: int = 2000,
        max_seq_len: int = 512,
    ) -> AccuracyResult:
        """
        Evaluate token accuracy on *data_path*.

        Returns
        -------
        AccuracyResult with accuracy, correct, total, eval_time_s.
        """
        dataset = _PerplexityDataset(data_path, self.tokenizer, max_seq_len, max_samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        all_refs: List[List[int]] = []
        all_preds: List[List[int]] = []
        t0 = time.time()

        for batch in loader:
            tokens = batch.to(self.device)            # (B, T+1)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]

            logits, _ = self.model(inputs)
            preds = logits.argmax(dim=-1)              # (B, T)

            all_refs.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        pad_id = self.model.config.pad_id
        acc_result = token_accuracy(all_refs, all_preds, ignore_id=pad_id)
        elapsed = time.time() - t0

        log.info(
            "Token accuracy — %.4f (%d/%d) | time: %.1fs",
            acc_result["accuracy"], acc_result["correct"], acc_result["total"], elapsed,
        )
        return AccuracyResult(
            accuracy=acc_result["accuracy"],
            correct=acc_result["correct"],
            total=acc_result["total"],
            eval_time_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Generation Quality Metrics
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class GenerationQualityResult:
    n_prompts: int
    avg_tokens: float
    avg_hindi_ratio: float
    avg_latin_ratio: float
    avg_distinct_1: float        # unigram diversity
    avg_distinct_2: float        # bigram diversity
    avg_repetition_rate: float   # fraction of repeated 4-grams
    corpus_bleu: float = 0.0     # corpus-level BLEU score
    corpus_chrf: float = 0.0     # corpus-level ChrF score
    samples: List[Dict] = field(default_factory=list)


class GenerationQualityMetrics:
    """
    Analyse the linguistic quality of model-generated text.

    Metrics
    -------
    hindi_ratio      : Fraction of Devanagari characters in the output.
    latin_ratio      : Fraction of Latin characters in the output.
    distinct-1/2     : Fraction of unique unigrams/bigrams (diversity proxy).
    repetition_rate  : Fraction of 4-grams that are duplicates (lower = better).
    """

    def __init__(
        self,
        model: IndicLLM,
        tokenizer: spm.SentencePieceProcessor,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _script_ratios(self, text: str) -> Tuple[float, float]:
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return 0.0, 0.0
        n = len(chars)
        h = sum(1 for c in chars if _DEVANAGARI_RE.match(c))
        l_ = sum(1 for c in chars if _LATIN_RE.match(c))
        return h / n, l_ / n

    def _distinct_n(self, tokens: List[str], n: int) -> float:
        ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    def _repetition_rate(self, tokens: List[str], n: int = 4) -> float:
        ngrams = [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(v - 1 for v in counts.values() if v > 1)
        return repeated / len(ngrams)

    @torch.no_grad()
    def evaluate(
        self,
        prompts_path: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> GenerationQualityResult:
        """
        Generate completions for each prompt in *prompts_path* and compute metrics.

        The prompts file should be a JSONL where each line has a ``"prompt"`` field.

        Returns
        -------
        GenerationQualityResult with aggregate and per-sample metrics.
        """
        prompts: List[str] = []
        with open(prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "prompt" in obj:
                        prompts.append(obj["prompt"])
                except Exception:
                    continue

        log.info("Evaluating generation quality on %d prompts …", len(prompts))

        all_hindi, all_latin, all_d1, all_d2, all_rep, all_ntok = [], [], [], [], [], []
        samples: List[Dict] = []

        eos_id = self.tokenizer.eos_id()

        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, out_type=int)
            input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)

            out = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_id=eos_id,
                use_kv_cache=False,
            )

            new_ids = out[0, len(ids):].tolist()
            generated_text = self.tokenizer.decode(new_ids)
            generated_pieces = self.tokenizer.encode(generated_text, out_type=str)

            h_ratio, l_ratio = self._script_ratios(generated_text)
            d1 = self._distinct_n(generated_pieces, 1)
            d2 = self._distinct_n(generated_pieces, 2)
            rep = self._repetition_rate(generated_pieces, 4)
            n_tok = len(new_ids)

            all_hindi.append(h_ratio)
            all_latin.append(l_ratio)
            all_d1.append(d1)
            all_d2.append(d2)
            all_rep.append(rep)
            all_ntok.append(n_tok)

            samples.append({
                "prompt": prompt,
                "generated": generated_text,
                "n_tokens": n_tok,
                "hindi_ratio": round(h_ratio, 4),
                "distinct_1": round(d1, 4),
                "distinct_2": round(d2, 4),
                "repetition_rate": round(rep, 4),
            })

        def _avg(lst: List[float]) -> float:
            return sum(lst) / max(1, len(lst))

        # Compute corpus-level BLEU and ChrF across all reference/hypothesis pairs
        ref_texts = [s["prompt"] for s in samples]
        hyp_texts = [s["generated"] for s in samples]
        bleu_result = compute_bleu(ref_texts, hyp_texts)
        chrf_result = compute_chrf(ref_texts, hyp_texts)

        result = GenerationQualityResult(
            n_prompts=len(prompts),
            avg_tokens=_avg(all_ntok),
            avg_hindi_ratio=_avg(all_hindi),
            avg_latin_ratio=_avg(all_latin),
            avg_distinct_1=_avg(all_d1),
            avg_distinct_2=_avg(all_d2),
            avg_repetition_rate=_avg(all_rep),
            corpus_bleu=bleu_result["bleu"],
            corpus_chrf=chrf_result["chrf"],
            samples=samples,
        )

        log.info(
            "Generation quality — hindi_ratio: %.3f | distinct-1: %.3f | "
            "distinct-2: %.3f | rep: %.3f | BLEU: %.4f | ChrF: %.4f",
            result.avg_hindi_ratio,
            result.avg_distinct_1,
            result.avg_distinct_2,
            result.avg_repetition_rate,
            result.corpus_bleu,
            result.corpus_chrf,
        )
        return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_eval(
    checkpoint_path: str,
    data_path: str,
    tokenizer_path: str,
    prompts_path: Optional[str] = None,
    output_path: str = "eval/results/report.json",
    device_str: str = "auto",
    batch_size: int = 8,
    max_samples: int = 2000,
) -> Dict:
    """
    Run the full evaluation suite and write a JSON report.

    Parameters
    ----------
    checkpoint_path : Path to a .pt checkpoint file.
    data_path       : JSONL corpus file for perplexity evaluation.
    tokenizer_path  : SentencePiece .model file.
    prompts_path    : Optional JSONL file with generation prompts.
    output_path     : Where to write the JSON report.
    device_str      : "auto", "cuda", "cpu".
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    model = _load_model_from_checkpoint(checkpoint_path, device)

    report: Dict = {
        "checkpoint": checkpoint_path,
        "device": str(device),
    }

    # Perplexity
    ppl_eval = PerplexityEvaluator(model, tokenizer, device)
    ppl_result = ppl_eval.evaluate(data_path, batch_size=batch_size, max_samples=max_samples)
    report["perplexity"] = asdict(ppl_result)

    # Token-level accuracy
    acc_eval = AccuracyEvaluator(model, tokenizer, device)
    acc_result = acc_eval.evaluate(data_path, batch_size=batch_size, max_samples=max_samples)
    report["token_accuracy"] = asdict(acc_result)

    # Generation quality (optional)
    if prompts_path and Path(prompts_path).exists():
        gen_eval = GenerationQualityMetrics(model, tokenizer, device)
        gen_result = gen_eval.evaluate(prompts_path)
        gen_dict = asdict(gen_result)
        gen_dict.pop("samples", None)   # omit verbose samples from top-level summary
        report["generation_quality"] = gen_dict
        report["generation_samples"] = gen_result.samples

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("Evaluation report saved → %s", output_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Indic LLM — evaluation harness",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data", required=True, help="JSONL corpus for perplexity eval")
    parser.add_argument("--tokenizer", default="data/tokenizer/indic_spm.model")
    parser.add_argument("--prompts", default=None,
                        help="JSONL file with generation prompts (optional)")
    parser.add_argument("--output", default="eval/results/report.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples for perplexity evaluation")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    report = run_eval(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        prompts_path=args.prompts,
        output_path=args.output,
        device_str=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "generation_samples"},
                     ensure_ascii=False, indent=2))

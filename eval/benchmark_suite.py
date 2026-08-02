"""
Indic LLM — Multi-Choice Evaluation & Benchmark Suite
======================================================
Implements rigorous sequence log-likelihood evaluation for multiple-choice
question answering (MCQA) on Indic benchmarks:

  - MCQASample          : Dataclass representing question, choices, and answer index
  - load_mcqa_dataset   : Loads structured JSONL benchmark datasets
  - compute_choice_logp : Computes length-normalized log-probabilities P(choice | prompt)
  - evaluate_mcqa       : Evaluates model across tasks and produces categorical breakdowns
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sentencepiece as spm
import torch
import torch.nn.functional as F

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from model import IndicLLM
except ImportError:
    from src.model import IndicLLM  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


@dataclass
class MCQASample:
    """A single multiple-choice question answering sample."""

    question: str
    options: List[str]
    answer_idx: int
    category: str = "General"
    context: str = ""


@dataclass
class PredictionResult:
    """Detailed evaluation result for a single question."""

    question: str
    category: str
    options: List[str]
    correct_idx: int
    predicted_idx: int
    choice_scores: List[float]
    is_correct: bool


@dataclass
class BenchmarkResult:
    """Aggregated benchmark evaluation metrics and breakdown."""

    total_samples: int
    correct_samples: int
    accuracy: float
    category_metrics: Dict[str, Dict[str, Union[int, float]]] = field(
        default_factory=dict
    )
    predictions: List[PredictionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to serializable dictionary."""
        return {
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "accuracy": self.accuracy,
            "category_metrics": self.category_metrics,
            "predictions": [asdict(p) for p in self.predictions],
        }


def load_mcqa_dataset(file_path: Union[str, Path]) -> List[MCQASample]:
    """
    Load MCQA benchmark samples from a JSONL file.

    Each line must contain:
      {"question": "...", "options": ["A", "B", ...], "answer_idx": 0, "category": "..."}
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    samples: List[MCQASample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(
                    MCQASample(
                        question=str(data["question"]),
                        options=[str(opt) for opt in data["options"]],
                        answer_idx=int(data["answer_idx"]),
                        category=str(data.get("category", "General")),
                        context=str(data.get("context", "")),
                    )
                )
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                log.warning("Skipping invalid sample at line %d: %s", line_num, e)

    log.info("Loaded %d MCQA samples from %s", len(samples), path.name)
    return samples


def format_mcqa_prompt(sample: MCQASample, style: str = "direct") -> str:
    """
    Format prompt string from an MCQASample.

    Styles:
      - 'direct' : "प्रश्न: {q}\\nउत्तर:"
      - 'context': "संदर्भ: {c}\\nप्रश्न: {q}\\nउत्तर:"
    """
    if style == "context" and sample.context:
        return f"संदर्भ: {sample.context}\nप्रश्न: {sample.question}\nउत्तर:"
    return f"प्रश्न: {sample.question}\nउत्तर:"


def compute_choice_logp(
    model: IndicLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    choice: str,
    device: torch.device,
    length_norm: bool = True,
) -> float:
    """
    Compute log P(choice | prompt) using token-level log-softmax.

    Parameters
    ----------
    model       : IndicLLM model
    tokenizer   : SentencePiece tokenizer
    prompt      : Condition prompt string
    choice      : Candidate answer string
    device      : Device to run computation on
    length_norm : If True, divide total log-probability by number of choice tokens

    Returns
    -------
    log_p       : Scalar log-likelihood score
    """
    prompt_tokens = tokenizer.encode(prompt, out_type=int)
    # Ensure BOS token is included if present in tokenizer
    bos_id = tokenizer.bos_id()
    if bos_id >= 0 and (not prompt_tokens or prompt_tokens[0] != bos_id):
        prompt_tokens = [bos_id] + prompt_tokens

    choice_str = (" " if not choice.startswith(" ") else "") + choice
    choice_tokens = tokenizer.encode(choice_str, out_type=int)

    if not choice_tokens:
        return float("-inf")

    full_tokens = prompt_tokens + choice_tokens
    t_prompt = len(prompt_tokens)
    t_choice = len(choice_tokens)

    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids)  # (1, seq_len, vocab_size)

        # Logits at index t predict token at index t + 1
        # Choice tokens are at indices [t_prompt, t_prompt + t_choice)
        # Therefore, logits of interest are [t_prompt - 1, t_prompt + t_choice - 1)
        choice_logits = logits[0, t_prompt - 1 : t_prompt + t_choice - 1, :]
        log_probs = F.log_softmax(choice_logits, dim=-1)

        target_ids = input_ids[0, t_prompt : t_prompt + t_choice]
        token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        sum_logp = token_logps.sum().item()

    if length_norm and t_choice > 0:
        return sum_logp / t_choice
    return sum_logp


def evaluate_mcqa(
    model: IndicLLM,
    tokenizer: spm.SentencePieceProcessor,
    samples_or_path: Union[List[MCQASample], str, Path],
    device: Optional[torch.device] = None,
    length_norm: bool = True,
    prompt_style: str = "direct",
) -> BenchmarkResult:
    """
    Evaluate IndicLLM on a set of multiple-choice questions.

    For each question, computes length-normalized log-likelihood for all options
    and selects argmax_{i} log P(option_i | prompt).
    """
    if isinstance(samples_or_path, (str, Path)):
        samples = load_mcqa_dataset(samples_or_path)
    else:
        samples = samples_or_path

    if not samples:
        return BenchmarkResult(
            total_samples=0,
            correct_samples=0,
            accuracy=0.0,
            category_metrics={},
            predictions=[],
        )

    if device is None:
        device = next(model.parameters()).device

    predictions: List[PredictionResult] = []
    category_stats: Dict[str, Dict[str, int]] = {}

    for sample in samples:
        prompt = format_mcqa_prompt(sample, style=prompt_style)
        scores: List[float] = []

        for opt in sample.options:
            score = compute_choice_logp(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                choice=opt,
                device=device,
                length_norm=length_norm,
            )
            scores.append(score)

        pred_idx = int(torch.tensor(scores).argmax().item())
        is_correct = pred_idx == sample.answer_idx

        # Accumulate category stats
        cat = sample.category
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1

        predictions.append(
            PredictionResult(
                question=sample.question,
                category=sample.category,
                options=sample.options,
                correct_idx=sample.answer_idx,
                predicted_idx=pred_idx,
                choice_scores=scores,
                is_correct=is_correct,
            )
        )

    total_samples = len(samples)
    correct_samples = sum(1 for p in predictions if p.is_correct)
    overall_acc = correct_samples / total_samples if total_samples > 0 else 0.0

    category_metrics: Dict[str, Dict[str, Union[int, float]]] = {}
    for cat, stats in category_stats.items():
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        category_metrics[cat] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(cat_acc * 100.0, 2),
        }

    log.info(
        "MCQA Evaluation Finished: %d/%d correct (%.2f%%)",
        correct_samples,
        total_samples,
        overall_acc * 100.0,
    )

    return BenchmarkResult(
        total_samples=total_samples,
        correct_samples=correct_samples,
        accuracy=round(overall_acc * 100.0, 2),
        category_metrics=category_metrics,
        predictions=predictions,
    )

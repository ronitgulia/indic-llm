"""
Indic LLM — Multi-Choice Evaluation & Benchmark Suite
======================================================
Implements rigorous sequence log-likelihood evaluation for multiple-choice
question answering (MCQA) on Indic benchmarks:

  - MCQASample               : Dataclass representing question, choices, and answer index
  - load_mcqa_dataset        : Loads structured JSONL benchmark datasets
  - compute_choice_logp      : Computes length-normalized log-probabilities P(choice | prompt)
  - evaluate_mcqa            : Evaluates model across tasks and produces categorical breakdowns
  - generate_markdown_report : Formats evaluation results into rich Markdown summaries
  - generate_leaderboard_table: Compiles cross-model comparative performance leaderboards
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
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
    from lora import LoRAConfig, apply_lora, load_adapter
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.lora import LoRAConfig, apply_lora, load_adapter  # type: ignore[no-redef]
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]

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


def generate_markdown_report(
    result: BenchmarkResult,
    model_name: str = "IndicLLM",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate an evaluation report in GitHub-Flavored Markdown format.
    Includes overall accuracy, categorical breakdowns, and sample prediction tables.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Indic LLM — Benchmark Evaluation Report",
        "",
        f"- **Model:** `{model_name}`",
        f"- **Timestamp:** `{timestamp}`",
        f"- **Total Evaluated:** {result.total_samples} questions",
        f"- **Overall Accuracy:** **{result.accuracy:.2f}%** ({result.correct_samples}/{result.total_samples})",
        "",
        "## Category Performance Breakdown",
        "",
        "| Category | Accuracy | Correct / Total |",
        "|:---------|:--------:|:---------------:|",
    ]

    for cat, metrics in sorted(result.category_metrics.items()):
        acc = metrics["accuracy"]
        correct = metrics["correct"]
        total = metrics["total"]
        lines.append(f"| {cat} | **{acc:.1f}%** | {correct} / {total} |")

    lines.extend([
        "",
        "## Sample Predictions & Diagnostics",
        "",
        "| # | Category | Question | Predicted | Ground Truth | Status |",
        "|:--|:---------|:---------|:----------|:-------------|:------:|",
    ])

    for i, pred in enumerate(result.predictions, 1):
        pred_choice = pred.options[pred.predicted_idx] if pred.predicted_idx < len(pred.options) else "N/A"
        gt_choice = pred.options[pred.correct_idx] if pred.correct_idx < len(pred.options) else "N/A"
        status = "✅ Correct" if pred.is_correct else "❌ Error"
        q_snippet = pred.question.replace("|", "/")
        lines.append(
            f"| {i} | {pred.category} | {q_snippet} | {pred_choice} | {gt_choice} | {status} |"
        )

    lines.append("")
    report_content = "\n".join(lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report_content)
        log.info("Saved Markdown benchmark report to %s", out)

    return report_content


def generate_leaderboard_table(
    results_map: Dict[str, BenchmarkResult],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Compile a comparative Markdown leaderboard across multiple models or checkpoints.
    """
    if not results_map:
        return "No results available for leaderboard."

    # Collect all unique categories
    all_categories: List[str] = sorted(
        {cat for res in results_map.values() for cat in res.category_metrics.keys()}
    )

    header = ["| Model | Overall Accuracy |"] + [f" {cat} |" for cat in all_categories]
    separator = ["|:------|:----------------:|"] + [":--------:|" for _ in all_categories]

    lines = [
        "## Indic LLM Benchmark Leaderboard",
        "",
        "".join(header),
        "".join(separator),
    ]

    for model_name, res in results_map.items():
        row = [f"| **{model_name}** | **{res.accuracy:.1f}%** |"]
        for cat in all_categories:
            cat_data = res.category_metrics.get(cat)
            acc_str = f"{cat_data['accuracy']:.1f}%" if cat_data else "—"
            row.append(f" {acc_str} |")
        lines.append("".join(row))

    lines.append("")
    content = "\n".join(lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("Saved Leaderboard table to %s", out)

    return content


def _load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    adapter_dir: Optional[str] = None,
) -> IndicLLM:
    """Load IndicLLM model from checkpoint with optional LoRA adapter."""
    log.info("Loading checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    raw_cfg = ckpt.get("config", {})
    model_cfg = ModelConfig(
        vocab_size=raw_cfg.get("vocab_size", 32000),
        dim=raw_cfg.get("model_dim", 512),
        n_layers=raw_cfg.get("n_layers", 8),
        n_heads=raw_cfg.get("n_heads", 8),
        n_kv_heads=raw_cfg.get("n_kv_heads", 4),
        max_seq_len=raw_cfg.get("max_seq_len", 512),
        dropout=0.0,
    )

    model = IndicLLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if adapter_dir and os.path.isdir(adapter_dir):
        log.info("Attaching LoRA adapter from %s", adapter_dir)
        cfg_path = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                lora_dict = json.load(f)
            lora_cfg = LoRAConfig(**lora_dict)
        else:
            lora_cfg = LoRAConfig()
        model = apply_lora(model, lora_cfg)
        load_adapter(model, adapter_dir)

    model.eval()
    return model


def main() -> None:
    """CLI entry point for benchmark evaluation and leaderboard reporting."""
    parser = argparse.ArgumentParser(
        description="Indic LLM Multi-Choice Evaluation & Benchmark Suite"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to SentencePiece model file (e.g. data/tokenizer/indic_spm.model)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="eval/tasks/indic_qa_samples.jsonl",
        help="Path to benchmark JSONL task dataset",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/benchmark_results.json",
        help="Path to write JSON benchmark metrics",
    )
    parser.add_argument(
        "--report_md",
        type=str,
        default="eval/results/benchmark_report.md",
        help="Path to write Markdown evaluation report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        choices=["direct", "context"],
        default="direct",
        help="Prompt formatting strategy",
    )
    parser.add_argument(
        "--no_length_norm",
        action="store_true",
        help="Disable length normalization in sequence log-probability computation",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Load tokenizer
    if not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer)

    # 2. Load model
    model = _load_model_from_checkpoint(
        ckpt_path=args.checkpoint,
        device=device,
        adapter_dir=args.adapter_dir,
    )

    # 3. Run evaluation
    results = evaluate_mcqa(
        model=model,
        tokenizer=tokenizer,
        samples_or_path=args.task,
        device=device,
        length_norm=not args.no_length_norm,
        prompt_style=args.prompt_style,
    )

    # 4. Save JSON results
    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
    log.info("Saved JSON benchmark results to %s", out_json)

    # 5. Generate Markdown report
    model_name = Path(args.checkpoint).stem
    if args.adapter_dir:
        model_name += " + LoRA"
    generate_markdown_report(results, model_name=model_name, output_path=args.report_md)


if __name__ == "__main__":
    main()

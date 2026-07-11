"""
Indic LLM — Supervised Fine-Tuning (SFT) Pipeline
===================================================
Instruction fine-tuning module with:
  - Target-only loss masking (prompts masked with IGNORE_INDEX = -100)
  - Chat template formatting ([INST] ... [/INST] and system prompt support)
  - AdamW optimizer with cosine LR schedule + linear warmup
  - Gradient accumulation and mixed-precision (AMP)
  - Checkpoint management with response-level validation metrics
  - YAML configuration loading and CLI overrides
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from model import IndicLLM, ModelConfig

# ─────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IGNORE_INDEX: int = -100


# ─────────────────────────────────────────
# SFT Configuration
# ─────────────────────────────────────────


@dataclass
class SFTConfig:
    # Model & Base weights
    base_checkpoint: Optional[str] = None
    tokenizer_path: str = "data/tokenizer/indic_spm.model"
    output_dir: str = "checkpoints/sft"

    # Dataset
    data_path: str = "data/processed/indic_instruct_clean.jsonl"
    max_seq_len: int = 512
    val_fraction: float = 0.05
    system_prompt: Optional[str] = None
    max_samples: Optional[int] = None

    # Model architecture (fallback if base_checkpoint not supplied)
    model_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 32000
    dropout: float = 0.05
    sliding_window: Optional[int] = None

    # Optimization
    batch_size: int = 4
    grad_accumulation_steps: int = 8  # Effective batch size = 32
    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 50
    max_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Logging & Checkpointing
    eval_every: int = 100
    save_every: int = 250
    keep_best_n: int = 3
    log_every: int = 20

    # Hardware & Environment
    device: str = "auto"
    use_amp: bool = True
    seed: int = 42
    num_workers: int = 0

    # Tracking
    use_wandb: bool = False
    wandb_project: str = "indic-llm-sft"
    wandb_run_name: Optional[str] = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accumulation_steps

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    @classmethod
    def from_yaml(cls, path: str) -> SFTConfig:
        """Load configuration from a YAML file."""
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load YAML configs: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        flat: Dict = {}
        for section in ("model", "sft", "data", "training", "checkpointing", "logging"):
            flat.update(raw.get(section, {}))

        key_map = {
            "dim": "model_dim",
            "lr": "learning_rate",
        }
        mapped = {key_map.get(k, k): v for k, v in flat.items()}
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in mapped.items() if k in valid_fields}
        return cls(**filtered)


# ─────────────────────────────────────────
# Dataset with Target-Only Loss Masking
# ─────────────────────────────────────────


class InstructDataset(Dataset):
    """
    Dataset for instruction fine-tuning that masks prompt tokens.

    For each instruction-response pair:
      Input:   [INST] {instruction} [/INST] {output} <eos>
      Targets: -100   ...           -100    {output} <eos>

    Loss is computed strictly on response tokens, teaching the model
    how to answer rather than memorising user prompts.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int = 512,
        system_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.examples: List[Tuple[str, str]] = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Instruction dataset not found: {data_path}")

        log.info("Loading instruction data from: %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(self.examples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    instruction, output = self._parse_record(obj)
                    if instruction and output:
                        self.examples.append((instruction, output))
                except Exception:
                    continue

        rng = random.Random(seed)
        rng.shuffle(self.examples)
        log.info("Loaded %d instruction samples", len(self.examples))

    def _parse_record(self, obj: dict) -> Tuple[Optional[str], Optional[str]]:
        if "instruction" in obj and "output" in obj:
            return obj["instruction"].strip(), obj["output"].strip()
        if "messages" in obj:
            user_msg = None
            asst_msg = None
            for m in obj["messages"]:
                if m.get("role") == "user":
                    user_msg = m.get("content", "").strip()
                elif m.get("role") == "assistant":
                    asst_msg = m.get("content", "").strip()
            return user_msg, asst_msg
        if "text" in obj:
            text = obj["text"]
            if "[INST]" in text and "[/INST]" in text:
                parts = text.split("[/INST]", 1)
                inst = parts[0].replace("[INST]", "").strip()
                out = parts[1].strip()
                return inst, out
        return None, None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instruction, output = self.examples[idx]

        # Format prompt
        if self.system_prompt:
            prompt_str = (
                f"<|system|> {self.system_prompt} "
                f"<|user|> {instruction} <|assistant|> "
            )
        else:
            prompt_str = f"[INST] {instruction} [/INST] "

        prompt_ids = self.tokenizer.encode(prompt_str, out_type=int)
        resp_ids = self.tokenizer.encode(output, out_type=int)

        eos_id = self.tokenizer.eos_id()
        if eos_id >= 0:
            resp_ids.append(eos_id)

        # Full sequence
        full_ids = prompt_ids + resp_ids
        prompt_len = len(prompt_ids)

        # Truncate if exceeds max_seq_len + 1
        if len(full_ids) > self.max_seq_len + 1:
            full_ids = full_ids[: self.max_seq_len + 1]

        # Standard causal shift: inputs = full_ids[:-1], targets = full_ids[1:]
        inputs = full_ids[:-1]
        targets = full_ids[1:]

        # Mask prompt tokens in target: tokens predicting prompt positions get IGNORE_INDEX
        # targets[i] = full_ids[i+1]. For i+1 < prompt_len, the target is inside the prompt.
        num_prompt_targets = max(0, prompt_len - 1)
        targets = [
            IGNORE_INDEX if i < num_prompt_targets else tok
            for i, tok in enumerate(targets)
        ]

        # Pad to max_seq_len
        pad_id = self.tokenizer.pad_id() if self.tokenizer.pad_id() >= 0 else 0
        pad_len = self.max_seq_len - len(inputs)
        if pad_len > 0:
            inputs = inputs + [pad_id] * pad_len
            targets = targets + [IGNORE_INDEX] * pad_len

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


# ─────────────────────────────────────────
# Learning Rate Scheduler
# ─────────────────────────────────────────


def get_cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────
# SFT Trainer
# ─────────────────────────────────────────


class SFTTrainer:
    """End-to-end instruction fine-tuning manager."""

    def __init__(self, cfg: SFTConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.resolved_device)

        # Set seeds for reproducibility
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # Load tokenizer
        if not os.path.exists(cfg.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {cfg.tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(cfg.tokenizer_path)
        log.info("Tokenizer loaded (vocab: %d)", self.tokenizer.get_piece_size())

        # Build or load model
        self.model = self._setup_model()

        # Build dataset & splits
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self._best_checkpoints: List[Tuple[float, str]] = []
        self._wandb_run = None

    def _setup_model(self) -> IndicLLM:
        """Initialize IndicLLM, restoring from checkpoint if provided."""
        if self.cfg.base_checkpoint and os.path.exists(self.cfg.base_checkpoint):
            log.info("Loading base weights: %s", self.cfg.base_checkpoint)
            ckpt = torch.load(self.cfg.base_checkpoint, map_location=self.device)
            raw_cfg = ckpt.get("config", {})
            model_cfg = ModelConfig(
                vocab_size=raw_cfg.get("vocab_size", self.cfg.vocab_size),
                dim=raw_cfg.get("model_dim", self.cfg.model_dim),
                n_layers=raw_cfg.get("n_layers", self.cfg.n_layers),
                n_heads=raw_cfg.get("n_heads", self.cfg.n_heads),
                n_kv_heads=raw_cfg.get("n_kv_heads", self.cfg.n_kv_heads),
                max_seq_len=self.cfg.max_seq_len,
                dropout=self.cfg.dropout,
                sliding_window=raw_cfg.get("sliding_window", self.cfg.sliding_window),
            )
            model = IndicLLM(model_cfg).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            log.info(
                "Restored base checkpoint (%.1fM params)",
                model.num_parameters() / 1e6,
            )
        else:
            if self.cfg.base_checkpoint:
                log.warning(
                    "Base checkpoint '%s' not found; training from scratch.",
                    self.cfg.base_checkpoint,
                )
            model_cfg = ModelConfig(
                vocab_size=self.cfg.vocab_size,
                dim=self.cfg.model_dim,
                n_layers=self.cfg.n_layers,
                n_heads=self.cfg.n_heads,
                n_kv_heads=self.cfg.n_kv_heads,
                max_seq_len=self.cfg.max_seq_len,
                dropout=self.cfg.dropout,
                sliding_window=self.cfg.sliding_window,
            )
            model = IndicLLM(model_cfg).to(self.device)
            log.info(
                "Initialized model from config (%.1fM params)",
                model.num_parameters() / 1e6,
            )

        return model

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation DataLoaders."""
        dataset = InstructDataset(
            data_path=self.cfg.data_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.cfg.max_seq_len,
            system_prompt=self.cfg.system_prompt,
            max_samples=self.cfg.max_samples,
            seed=self.cfg.seed,
        )

        n_val = max(1, int(len(dataset) * self.cfg.val_fraction))
        n_train = len(dataset) - n_val

        generator = torch.Generator().manual_seed(self.cfg.seed)
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val], generator=generator
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.num_workers,
        )
        log.info("Splits created: %d train | %d val", len(train_ds), len(val_ds))
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> Tuple[float, float, float]:
        """
        Evaluate masked validation loss, perplexity, and token accuracy.
        Returns: (avg_loss, perplexity, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        total_correct = 0
        total_tokens = 0

        for inputs, targets in self.val_loader:
            if n_batches >= max_batches:
                break
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits, loss = self.model(
                inputs, targets, ignore_index=IGNORE_INDEX
            )
            if loss is not None:
                total_loss += loss.item()
                n_batches += 1

            # Compute masked accuracy on response tokens
            preds = logits.argmax(dim=-1)
            valid_mask = targets != IGNORE_INDEX
            total_correct += ((preds == targets) & valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()

        self.model.train()
        avg_loss = total_loss / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20.0))
        acc = total_correct / max(1, total_tokens)
        return avg_loss, ppl, acc

    def _save_checkpoint(self, step: int, val_loss: float) -> str:
        """Save SFT checkpoint and keep best N by validation loss."""
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.output_dir, f"sft_step_{step:06d}.pt")

        payload = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": asdict(self.cfg),
        }
        torch.save(payload, ckpt_path)

        meta_path = ckpt_path.replace(".pt", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {"step": step, "val_loss": val_loss, "path": ckpt_path},
                mf,
                indent=2,
            )

        self._best_checkpoints.append((val_loss, ckpt_path))
        self._best_checkpoints.sort(key=lambda x: x[0])

        while len(self._best_checkpoints) > self.cfg.keep_best_n:
            _, old_path = self._best_checkpoints.pop()
            for ext in (".pt", "_meta.json"):
                p = old_path.replace(".pt", ext)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        log.info("Saved SFT checkpoint: %s (val_loss=%.4f)", ckpt_path, val_loss)
        return ckpt_path

    def train(self) -> None:
        """Run the main SFT training loop."""
        cfg = self.cfg
        log.info(
            "Starting SFT: %d steps | batch=%d (effective=%d) | lr=%.2e | device=%s",
            cfg.max_steps,
            cfg.batch_size,
            cfg.effective_batch_size,
            cfg.learning_rate,
            self.device,
        )

        use_amp = cfg.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if cfg.use_wandb and _WANDB_AVAILABLE:
            self._wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or "sft-run",
                config=asdict(cfg),
            )

        self.model.train()
        step = 0
        micro_step = 0
        running_loss = 0.0
        best_val_loss = float("inf")
        t_start = time.time()

        train_iter = iter(self.train_loader)

        while step < cfg.max_steps:
            # Advance iterator, cycling across epochs seamlessly
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                inputs, targets = next(train_iter)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            lr = get_cosine_schedule_with_warmup(
                step=step,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
                base_lr=cfg.learning_rate,
                min_lr=cfg.min_lr,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass under autocast
            with torch.amp.autocast(self.device.type, enabled=use_amp):
                _, loss = self.model(
                    inputs, targets, ignore_index=IGNORE_INDEX
                )
                loss_scaled = loss / cfg.grad_accumulation_steps

            scaler.scale(loss_scaled).backward()
            running_loss += loss.item()
            micro_step += 1

            # Optimizer step on gradient accumulation boundary
            if micro_step % cfg.grad_accumulation_steps == 0:
                scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                ).item()

                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                step += 1

                # Logging
                if step % cfg.log_every == 0:
                    avg_loss = running_loss / (
                        cfg.log_every * cfg.grad_accumulation_steps
                    )
                    ppl = math.exp(min(avg_loss, 20.0))
                    elapsed = time.time() - t_start
                    log.info(
                        "step %4d/%d | loss %.4f | ppl %6.2f | lr %.2e | gnorm %.3f | time %.1fs",
                        step,
                        cfg.max_steps,
                        avg_loss,
                        ppl,
                        lr,
                        grad_norm,
                        elapsed,
                    )
                    if self._wandb_run:
                        self._wandb_run.log(
                            {
                                "sft/loss": avg_loss,
                                "sft/ppl": ppl,
                                "sft/lr": lr,
                                "sft/grad_norm": grad_norm,
                                "step": step,
                            }
                        )
                    running_loss = 0.0

                # Validation & Checkpointing
                if step % cfg.eval_every == 0 or step == cfg.max_steps:
                    val_loss, val_ppl, val_acc = self.evaluate()
                    log.info(
                        "  [Validation] step %d: loss=%.4f | ppl=%.2f | token_acc=%.2f%%",
                        step,
                        val_loss,
                        val_ppl,
                        val_acc * 100.0,
                    )
                    if self._wandb_run:
                        self._wandb_run.log(
                            {
                                "val/loss": val_loss,
                                "val/ppl": val_ppl,
                                "val/token_accuracy": val_acc,
                                "step": step,
                            }
                        )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(cfg.output_dir, "sft_best.pt")
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": self.model.state_dict(),
                                "val_loss": val_loss,
                                "config": asdict(cfg),
                            },
                            best_path,
                        )
                        log.info("  ★ New best SFT model saved: %s", best_path)

                if step % cfg.save_every == 0:
                    self._save_checkpoint(step, val_loss=running_loss)

        total_time = (time.time() - t_start) / 60.0
        log.info("SFT complete in %.1f minutes. Best val loss: %.4f", total_time, best_val_loss)
        if self._wandb_run:
            self._wandb_run.finish()


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indic LLM — Supervised Fine-Tuning (SFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--base_checkpoint", type=str, default=None, help="Path to pretrained base checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to instruction JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save SFT checkpoints"
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum SFT steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--use_wandb", action="store_true", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    if cli_args.config:
        cfg = SFTConfig.from_yaml(cli_args.config)
    else:
        cfg = SFTConfig()

    # Apply non-None CLI overrides
    for k, v in vars(cli_args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    trainer = SFTTrainer(cfg)
    trainer.train()

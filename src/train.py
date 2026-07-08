"""
Indic LLM - Training Pipeline
==============================
Full training loop with:
  - AdamW optimizer + cosine LR schedule with warmup
  - Gradient clipping & accumulation
  - Mixed precision (torch.autocast)
  - Checkpoint save/resume
  - JSONL dataset loading for Indic corpora
  - Training metrics logging (loss, perplexity, tokens/sec)
"""

import os
import time
import math
import json
import random
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from model import IndicLLM, ModelConfig

# ─────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  Training Config
# ─────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    data_paths: List[str] = field(default_factory=lambda: [
        "data/processed/sangraha_clean.jsonl",
        "data/processed/hindi_parallel_clean.jsonl",
        "data/processed/indic_instruct_clean.jsonl",
    ])
    tokenizer_path: str = "data/tokenizer/indic_spm.model"
    max_seq_len: int = 512

    # Model
    model_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 32000

    # Training
    batch_size: int = 8
    grad_accumulation_steps: int = 4        # effective batch = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500                   # steps
    log_every: int = 50

    # Hardware
    device: str = "auto"
    use_amp: bool = True                    # mixed precision
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accumulation_steps

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ─────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────

class IndicTextDataset(Dataset):
    """
    Loads JSONL files from the processed Indic corpora.
    Tokenizes on-the-fly and creates fixed-length chunks.
    """

    def __init__(self, paths: List[str], tokenizer: spm.SentencePieceProcessor,
                 max_seq_len: int, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[str] = []

        for path in paths:
            if not os.path.exists(path):
                log.warning(f"Data file not found, skipping: {path}")
                continue

            log.info(f"Loading: {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = self._extract_text(obj)
                        if text and len(text) > 20:
                            self.samples.append(text)
                    except Exception:
                        continue

        random.shuffle(self.samples)

        if max_samples:
            self.samples = self.samples[:max_samples]

        log.info(f"Dataset loaded: {len(self.samples):,} samples")

    def _extract_text(self, obj: dict) -> Optional[str]:
        """Extract text from different JSONL formats."""
        if "text" in obj:
            return obj["text"]
        elif "hi" in obj:
            hi = obj["hi"]
            en = obj.get("en", "")
            return f"{hi} {en}".strip() if en else hi
        elif "instruction" in obj:
            instruction = obj.get("instruction", "")
            output = obj.get("output", "")
            return f"[INST] {instruction} [/INST] {output}"
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.samples[idx]
        ids = self.tokenizer.encode(text, out_type=int)

        # Pad or truncate to max_seq_len + 1 (for shifted targets)
        ids = ids[: self.max_seq_len + 1]
        if len(ids) < self.max_seq_len + 1:
            ids += [0] * (self.max_seq_len + 1 - len(ids))   # pad_id = 0

        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]):
    """Stack samples; return (inputs, targets) with causal shift."""
    tokens = torch.stack(batch)            # (B, T+1)
    inputs = tokens[:, :-1]               # (B, T)  — model input
    targets = tokens[:, 1:]               # (B, T)  — next-token targets
    return inputs, targets


# ─────────────────────────────────────────
#  LR Scheduler — cosine with warmup
# ─────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    # Linear warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    # Cosine decay after warmup
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────
#  Checkpoint utilities
# ─────────────────────────────────────────

def save_checkpoint(model: IndicLLM, optimizer: torch.optim.Optimizer,
                    step: int, loss: float, cfg: TrainConfig):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": cfg.__dict__,
    }, ckpt_path)
    log.info(f"Checkpoint saved → {ckpt_path}")


def load_checkpoint(ckpt_path: str, model: IndicLLM,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: str = "cpu") -> int:
    """Load checkpoint and return the step to resume from."""
    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt.get("step", 0)
    loss = ckpt.get("loss", float("inf"))
    log.info(f"Resumed from step {step}, last loss = {loss:.4f}")
    return step


# ─────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: TrainConfig, resume_from: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.resolved_device)

        self._set_seed(cfg.seed)
        log.info(f"Training on device: {self.device}")

        # Tokenizer
        log.info(f"Loading tokenizer: {cfg.tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(cfg.tokenizer_path)

        # Dataset & DataLoader
        dataset = IndicTextDataset(
            paths=cfg.data_paths,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.data_iter = iter(self.loader)

        # Model
        model_cfg = ModelConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.model_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        self.model = IndicLLM(model_cfg).to(self.device)
        n_params = self.model.num_parameters()
        log.info(f"Model parameters: {n_params / 1e6:.2f}M")

        # Optimizer
        # Separate weight decay: don't decay embeddings, norms, biases
        decay_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and self.device.type == "cuda"))

        # Resume if checkpoint provided
        self.start_step = 0
        if resume_from and os.path.exists(resume_from):
            self.start_step = load_checkpoint(resume_from, self.model, self.optimizer, str(self.device))

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_batch(self):
        """Infinite data iterator — restarts when exhausted."""
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            return next(self.data_iter)

    def train(self):
        """Main training loop."""
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler

        log.info(f"\n{'='*60}")
        log.info(f"  Starting Indic LLM Training")
        log.info(f"  Steps        : {cfg.max_steps}")
        log.info(f"  Batch size   : {cfg.batch_size} × {cfg.grad_accumulation_steps} acc = {cfg.effective_batch_size}")
        log.info(f"  Learning rate: {cfg.learning_rate}")
        log.info(f"  Seq length   : {cfg.max_seq_len}")
        log.info(f"{'='*60}\n")

        model.train()
        optimizer.zero_grad()

        running_loss = 0.0
        tokens_seen = 0
        t_start = time.time()

        for step in range(self.start_step, cfg.max_steps):
            # Update LR
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            step_loss = 0.0
            for micro_step in range(cfg.grad_accumulation_steps):
                inputs, targets = self._get_batch()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.float16,
                                    enabled=(cfg.use_amp and self.device.type == "cuda")):
                    _, loss = model(inputs, targets)
                    loss = loss / cfg.grad_accumulation_steps

                scaler.scale(loss).backward()
                step_loss += loss.item()

                tokens_seen += inputs.numel()

            # Gradient clip + optimizer step
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += step_loss

            # ── Logging ──────────────────────────────────
            if (step + 1) % cfg.log_every == 0:
                elapsed = time.time() - t_start
                avg_loss = running_loss / cfg.log_every
                perplexity = math.exp(min(avg_loss, 20))
                tok_per_sec = tokens_seen / elapsed

                log.info(
                    f"step {step+1:>6}/{cfg.max_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"ppl {perplexity:.2f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tok_per_sec:,.0f}"
                )
                running_loss = 0.0

            # ── Checkpoint ───────────────────────────────
            if (step + 1) % cfg.save_every == 0:
                save_checkpoint(model, optimizer, step + 1, step_loss, cfg)

        # Final checkpoint
        save_checkpoint(model, optimizer, cfg.max_steps, step_loss, cfg)
        elapsed = time.time() - t_start
        log.info(f"\n Training complete in {elapsed/60:.1f} min")
        log.info(f" Total tokens seen: {tokens_seen:,}")


# ─────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Indic LLM")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        grad_accumulation_steps=args.grad_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_amp=not args.no_amp,
    )

    trainer = Trainer(cfg, resume_from=args.resume_from)
    trainer.train()

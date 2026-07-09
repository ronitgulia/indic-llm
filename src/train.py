"""
Indic LLM - Training Pipeline
==============================
Full training loop with:
  - AdamW optimizer + cosine LR schedule with linear warmup
  - Gradient clipping & accumulation
  - Mixed precision (torch.autocast)
  - Checkpoint save/resume with best-N checkpoint management
  - JSONL dataset loading for Indic corpora
  - Training metrics logging (loss, perplexity, tokens/sec, grad norm)
  - Optional Weights & Biases integration (--use_wandb)
  - Validation split evaluation
  - YAML config file loading (--config configs/train_config.yaml)
"""

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

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
    # ── Data ───────────────────────────────────────────────────────────
    data_paths: List[str] = field(default_factory=lambda: [
        "data/processed/sangraha_clean.jsonl",
        "data/processed/hindi_parallel_clean.jsonl",
        "data/processed/indic_instruct_clean.jsonl",
    ])
    tokenizer_path: str = "data/tokenizer/indic_spm.model"
    max_seq_len: int = 512
    val_fraction: float = 0.005     # fraction of data held out for validation

    # ── Model ──────────────────────────────────────────────────────────
    model_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 32000
    use_flash_attn: bool = False
    rope_scaling_factor: float = 1.0

    # ── Optimisation ───────────────────────────────────────────────────
    batch_size: int = 8
    grad_accumulation_steps: int = 4        # effective batch = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1

    # ── Checkpointing ──────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500                   # steps
    keep_best_n: int = 3                    # keep only the N best checkpoints
    log_every: int = 50
    eval_every: int = 500                   # run validation every N steps

    # ── Hardware ───────────────────────────────────────────────────────
    device: str = "auto"
    use_amp: bool = True                    # mixed precision
    seed: int = 42
    num_workers: int = 0

    # ── Logging integrations ───────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "indic-llm"
    wandb_run_name: Optional[str] = None
    experiment_name: str = "default"

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
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load a TrainConfig from a YAML file, merging with dataclass defaults."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Flatten nested YAML structure to dataclass fields
        flat: Dict = {}
        for section in ("model", "training", "data", "checkpointing", "logging"):
            flat.update(raw.get(section, {}))

        # Map YAML keys → dataclass fields
        key_map = {
            "dim": "model_dim",
            "paths": "data_paths",
            "tokenizer_path": "tokenizer_path",
            "checkpoint_dir": "checkpoint_dir",
            "save_every": "save_every",
            "log_every": "log_every",
            "use_wandb": "use_wandb",
        }
        mapped = {key_map.get(k, k): v for k, v in flat.items()}

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in mapped.items() if k in known_fields}
        return cls(**filtered)


# ─────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────

class IndicTextDataset(Dataset):
    """
    Loads JSONL files from the processed Indic corpora.
    Tokenises text on-the-fly and creates fixed-length (seq_len+1) chunks.
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[str] = []

        for path in paths:
            if not os.path.exists(path):
                log.warning("Data file not found, skipping: %s", path)
                continue
            log.info("Loading: %s", path)
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

        log.info("Dataset loaded: %d samples", len(self.samples))

    def _extract_text(self, obj: dict) -> Optional[str]:
        """Extract text from different JSONL formats."""
        if "text" in obj:
            return obj["text"]
        elif "hi" in obj:
            hi, en = obj["hi"], obj.get("en", "")
            return f"{hi} {en}".strip() if en else hi
        elif "instruction" in obj:
            return f"[INST] {obj.get('instruction', '')} [/INST] {obj.get('output', '')}"
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.samples[idx]
        ids = self.tokenizer.encode(text, out_type=int)
        ids = ids[: self.max_seq_len + 1]
        ids += [0] * (self.max_seq_len + 1 - len(ids))   # pad_id = 0
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack samples; return (inputs, targets) with causal one-position shift."""
    tokens = torch.stack(batch)       # (B, T+1)
    return tokens[:, :-1], tokens[:, 1:]


# ─────────────────────────────────────────
#  LR Scheduler — cosine with warmup
# ─────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────
#  Checkpoint utilities
# ─────────────────────────────────────────

def save_checkpoint(
    model: IndicLLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    cfg: TrainConfig,
    best_checkpoints: List[Tuple[float, str]],
) -> None:
    """
    Save a checkpoint and maintain a list of the best-N checkpoints.
    Older checkpoints that fall outside the top-N are deleted.
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"step_{step:06d}.pt")

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": asdict(cfg),
        },
        ckpt_path,
    )

    # Save metadata JSON alongside checkpoint for quick inspection
    meta_path = ckpt_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as mf:
        json.dump({"step": step, "val_loss": val_loss, "path": ckpt_path}, mf, indent=2)

    best_checkpoints.append((val_loss, ckpt_path))
    best_checkpoints.sort(key=lambda x: x[0])          # sort ascending by loss

    # Prune checkpoints beyond keep_best_n
    while len(best_checkpoints) > cfg.keep_best_n:
        _, old_path = best_checkpoints.pop()             # remove worst
        for ext in (".pt", "_meta.json"):
            try:
                os.remove(old_path.replace(".pt", ext))
                log.info("Pruned checkpoint: %s", old_path)
            except FileNotFoundError:
                pass

    log.info("Checkpoint saved → %s (val_loss=%.4f)", ckpt_path, val_loss)


def load_checkpoint(
    ckpt_path: str,
    model: IndicLLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """Load checkpoint and return the step to resume from."""
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    log.info("Resumed from step %d | val_loss=%.4f", step, val_loss)
    return step


# ─────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: IndicLLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Tuple[float, float]:
    """
    Evaluate cross-entropy loss and perplexity on a held-out DataLoader.

    Returns
    -------
    (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for inputs, targets in loader:
        if n_batches >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return avg_loss, perplexity


# ─────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: TrainConfig, resume_from: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.resolved_device)
        self._set_seed(cfg.seed)
        self._best_checkpoints: List[Tuple[float, str]] = []

        log.info("Training on: %s", self.device)

        # ── Tokeniser ──────────────────────────────────────────────────
        log.info("Loading tokenizer: %s", cfg.tokenizer_path)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(cfg.tokenizer_path)

        # ── Dataset & DataLoader ───────────────────────────────────────
        full_dataset = IndicTextDataset(
            paths=cfg.data_paths,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
        )

        val_size = max(1, int(len(full_dataset) * cfg.val_fraction))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
        )
        self._train_iter: Iterator = iter(self.train_loader)

        # ── Model ──────────────────────────────────────────────────────
        model_cfg = ModelConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.model_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            use_flash_attn=cfg.use_flash_attn,
            rope_scaling_factor=cfg.rope_scaling_factor,
        )
        self.model = IndicLLM(model_cfg).to(self.device)
        log.info("Model parameters: %.2fM", self.model.num_parameters() / 1e6)
        log.info("\n%s", self.model.model_summary())

        # ── Optimiser ──────────────────────────────────────────────────
        # Separate weight-decay groups: don't decay norms, embeddings, biases
        decay = [p for n, p in self.model.named_parameters()
                 if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.model.named_parameters()
                    if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # ── AMP scaler ─────────────────────────────────────────────────
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.use_amp and self.device.type == "cuda")
        )

        # ── Resume ─────────────────────────────────────────────────────
        self.start_step = 0
        if resume_from and os.path.exists(resume_from):
            self.start_step = load_checkpoint(
                resume_from, self.model, self.optimizer, str(self.device)
            )

        # ── W&B ────────────────────────────────────────────────────────
        self._wandb_run = None
        if cfg.use_wandb:
            if not _WANDB_AVAILABLE:
                log.warning("wandb not installed — skipping W&B logging (pip install wandb)")
            else:
                self._wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name or cfg.experiment_name,
                    config=asdict(cfg),
                    resume="allow",
                )
                log.info("Weights & Biases run: %s", self._wandb_run.url)

    # ──────────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infinite data iterator — restarts when exhausted."""
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    # ──────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Main training loop."""
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler

        log.info("\n%s", "=" * 60)
        log.info("  Indic LLM Training")
        log.info("  Steps:          %d", cfg.max_steps)
        log.info("  Effective batch: %d (%d × %d acc)", cfg.effective_batch_size,
                 cfg.batch_size, cfg.grad_accumulation_steps)
        log.info("  Learning rate:  %.2e → %.2e (cosine)", cfg.learning_rate, cfg.min_lr)
        log.info("  Seq length:     %d", cfg.max_seq_len)
        log.info("  Device:         %s", self.device)
        log.info("%s\n", "=" * 60)

        model.train()
        optimizer.zero_grad()

        running_loss = 0.0
        tokens_seen = 0
        t_start = time.time()
        best_val_loss = float("inf")

        for step in range(self.start_step, cfg.max_steps):
            # ── LR update ────────────────────────────────────────────────
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Gradient accumulation ────────────────────────────────────
            step_loss = 0.0
            for _ in range(cfg.grad_accumulation_steps):
                inputs, targets = self._get_batch()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=(cfg.use_amp and self.device.type == "cuda"),
                ):
                    _, loss = model(inputs, targets)
                    loss = loss / cfg.grad_accumulation_steps

                scaler.scale(loss).backward()
                step_loss += loss.item()
                tokens_seen += inputs.numel()

            # ── Grad clip + step ─────────────────────────────────────────
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += step_loss

            # ── Logging ──────────────────────────────────────────────────
            if (step + 1) % cfg.log_every == 0:
                elapsed = time.time() - t_start
                avg_loss = running_loss / cfg.log_every
                perplexity = math.exp(min(avg_loss, 20))
                tok_per_sec = tokens_seen / elapsed

                log.info(
                    "step %6d/%d | loss %.4f | ppl %7.2f | lr %.2e | "
                    "gnorm %.3f | tok/s %8,.0f",
                    step + 1, cfg.max_steps, avg_loss, perplexity,
                    lr, grad_norm, tok_per_sec,
                )

                if self._wandb_run is not None:
                    self._wandb_run.log({
                        "train/loss": avg_loss,
                        "train/perplexity": perplexity,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tok_per_sec,
                        "step": step + 1,
                    })

                running_loss = 0.0

            # ── Validation + Checkpoint ───────────────────────────────────
            if (step + 1) % cfg.eval_every == 0:
                val_loss, val_ppl = evaluate(model, self.val_loader, self.device)
                log.info(
                    "  ── val │ loss %.4f │ ppl %.2f │ best %.4f",
                    val_loss, val_ppl, best_val_loss,
                )

                if self._wandb_run is not None:
                    self._wandb_run.log({
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "step": step + 1,
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                save_checkpoint(
                    model, optimizer, step + 1, val_loss, cfg, self._best_checkpoints
                )

        # Final validation + checkpoint
        val_loss, val_ppl = evaluate(model, self.val_loader, self.device)
        save_checkpoint(
            model, optimizer, cfg.max_steps, val_loss, cfg, self._best_checkpoints
        )

        elapsed = time.time() - t_start
        log.info("\n  Training complete in %.1f min", elapsed / 60)
        log.info("  Total tokens seen: %d", tokens_seen)
        log.info("  Best val loss: %.4f", best_val_loss)

        if self._wandb_run is not None:
            self._wandb_run.finish()


# ─────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indic LLM — training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides CLI defaults)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--keep_best_n", type=int, default=3,
                        help="Number of best checkpoints to retain")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .pt checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="indic-llm")
    parser.add_argument("--experiment_name", type=str, default="default",
                        help="Run name for checkpointing and W&B")
    parser.add_argument("--val_fraction", type=float, default=0.005,
                        help="Fraction of data to hold out for validation")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate validation loss every N steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config first (if provided), then override with CLI flags
    if args.config:
        cfg = TrainConfig.from_yaml(args.config)
        log.info("Config loaded from: %s", args.config)
    else:
        cfg = TrainConfig()

    # CLI flags take priority over YAML
    cfg.batch_size = args.batch_size
    cfg.max_steps = args.max_steps
    cfg.learning_rate = args.learning_rate
    cfg.max_seq_len = args.max_seq_len
    cfg.grad_accumulation_steps = args.grad_accumulation_steps
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.keep_best_n = args.keep_best_n
    cfg.device = args.device
    cfg.use_amp = not args.no_amp
    cfg.use_wandb = args.use_wandb
    cfg.wandb_project = args.wandb_project
    cfg.experiment_name = args.experiment_name
    cfg.val_fraction = args.val_fraction
    cfg.eval_every = args.eval_every

    trainer = Trainer(cfg, resume_from=args.resume_from)
    trainer.train()

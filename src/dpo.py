"""
Indic LLM — Direct Preference Optimization (DPO) Alignment Pipeline
=====================================================================
Post-training preference alignment using Direct Preference Optimization:
  - Eliminates the need for a separate reward model or PPO reinforcement learning.
  - Formats paired preference tuples: (prompt, chosen, rejected).
  - Target-only loss masking for chosen and rejected completions.
  - Calculates implicit policy and reference log-probabilities.
  - Optimizes the Bradley-Terry preference objective with margin logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Ensure src/ is importable
sys_path = str(Path(__file__).resolve().parent)
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)

try:
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dpo")

IGNORE_INDEX: int = -100


# ─────────────────────────────────────────
# Preference Dataset with Loss Masking
# ─────────────────────────────────────────


class PreferenceDataset(Dataset):
    """
    Dataset for paired preference feedback (prompt, chosen, rejected).

    Masks all prompt tokens in both chosen and rejected target sequences
    with IGNORE_INDEX (-100) so log-probabilities are strictly computed
    over response tokens.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[str, str, str]] = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Preference dataset not found: {data_path}")

        log.info("Loading preference dataset from: %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    prompt = obj.get("prompt") or obj.get("instruction") or ""
                    chosen = obj.get("chosen") or obj.get("accepted") or ""
                    rejected = obj.get("rejected") or ""
                    if prompt and chosen and rejected:
                        self.samples.append((prompt.strip(), chosen.strip(), rejected.strip()))
                except Exception:
                    continue

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        log.info("Loaded %d preference pairs", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_sequence(
        self, prompt: str, response: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_str = f"[INST] {prompt} [/INST] "
        prompt_ids = self.tokenizer.encode(prompt_str, out_type=int)
        resp_ids = self.tokenizer.encode(response, out_type=int)

        eos_id = self.tokenizer.eos_id()
        if eos_id >= 0:
            resp_ids.append(eos_id)

        full_ids = prompt_ids + resp_ids
        prompt_len = len(prompt_ids)

        # Truncate if exceeds max_seq_len + 1
        if len(full_ids) > self.max_seq_len + 1:
            full_ids = full_ids[: self.max_seq_len + 1]

        inputs = full_ids[:-1]
        targets = full_ids[1:]

        # Mask prompt tokens in target
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

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt, chosen, rejected = self.samples[idx]
        chosen_inputs, chosen_targets = self._encode_sequence(prompt, chosen)
        rejected_inputs, rejected_targets = self._encode_sequence(prompt, rejected)

        return chosen_inputs, chosen_targets, rejected_inputs, rejected_targets


# ─────────────────────────────────────────
# DPO Math & Loss Functions
# ─────────────────────────────────────────


def compute_sequence_logps(
    model: IndicLLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Compute total log-probability of target response tokens.

    Parameters
    ----------
    model        : IndicLLM model
    inputs       : (B, T) input token IDs
    targets      : (B, T) target token IDs with prompt tokens masked as ignore_index

    Returns
    -------
    log_probs    : (B,) sum of log-probabilities over valid response tokens
    """
    logits, _ = model(inputs)  # (B, T, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)

    # Replace ignore_index with 0 temporarily for gather
    mask = targets != ignore_index
    safe_targets = targets.clone()
    safe_targets[~mask] = 0

    per_token_logps = torch.gather(
        log_probs, dim=-1, index=safe_targets.unsqueeze(-1)
    ).squeeze(-1)

    # Zero out masked tokens
    per_token_logps = per_token_logps * mask.float()
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direct Preference Optimization (DPO) loss calculation.

    Parameters
    ----------
    policy_chosen_logps   : (B,) log p_theta(y_w | x)
    policy_rejected_logps : (B,) log p_theta(y_l | x)
    ref_chosen_logps      : (B,) log p_ref(y_w | x)
    ref_rejected_logps    : (B,) log p_ref(y_l | x)
    beta                  : DPO temperature hyperparameter (typical range 0.05 - 0.2)

    Returns
    -------
    loss                  : Scalar loss
    chosen_rewards        : (B,) implicit reward for chosen response
    rejected_rewards      : (B,) implicit reward for rejected response
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


# ─────────────────────────────────────────
# DPO Configuration
# ─────────────────────────────────────────


@dataclass
class DPOConfig:
    # Model checkpoints
    base_checkpoint: Optional[str] = None
    tokenizer_path: str = "data/tokenizer/indic_spm.model"
    output_dir: str = "checkpoints/dpo"

    # Dataset
    data_path: str = "data/processed/indic_preferences.jsonl"
    max_seq_len: int = 512
    val_fraction: float = 0.05
    max_samples: Optional[int] = None

    # Model architecture (if training from random weights)
    model_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 32000

    # Hyperparameters
    beta: float = 0.1
    batch_size: int = 2
    grad_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 1e-6       # DPO requires small LR
    min_lr: float = 1e-7
    warmup_steps: int = 20
    max_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Logging & Checkpointing
    eval_every: int = 50
    save_every: int = 100
    keep_best_n: int = 3
    log_every: int = 10

    # Environment
    device: str = "auto"
    use_amp: bool = True
    seed: int = 42

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_yaml(cls, path: str) -> DPOConfig:
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        flat: Dict = {}
        for section in ("model", "dpo", "data", "training", "checkpointing"):
            flat.update(raw.get(section, {}))

        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)


# ─────────────────────────────────────────
# DPO Trainer
# ─────────────────────────────────────────


class DPOTrainer:
    """Manages policy vs reference model alignment via DPO."""

    def __init__(self, cfg: DPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.resolved_device)

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(cfg.tokenizer_path)

        # Policy model (trainable)
        self.policy_model = self._load_or_build_model()
        self.policy_model.train()

        # Reference model (frozen)
        self.ref_model = self._load_or_build_model()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader, self.val_loader = self._build_dataloaders()
        self._best_margin: float = float("-inf")

    def _load_or_build_model(self) -> IndicLLM:
        if self.cfg.base_checkpoint and os.path.exists(self.cfg.base_checkpoint):
            ckpt = torch.load(self.cfg.base_checkpoint, map_location=self.device)
            raw_cfg = ckpt.get("config", {})
            model_cfg = ModelConfig(
                vocab_size=raw_cfg.get("vocab_size", self.cfg.vocab_size),
                dim=raw_cfg.get("model_dim", self.cfg.model_dim),
                n_layers=raw_cfg.get("n_layers", self.cfg.n_layers),
                n_heads=raw_cfg.get("n_heads", self.cfg.n_heads),
                n_kv_heads=raw_cfg.get("n_kv_heads", self.cfg.n_kv_heads),
                max_seq_len=self.cfg.max_seq_len,
            )
            model = IndicLLM(model_cfg).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model_cfg = ModelConfig(
                vocab_size=self.cfg.vocab_size,
                dim=self.cfg.model_dim,
                n_layers=self.cfg.n_layers,
                n_heads=self.cfg.n_heads,
                n_kv_heads=self.cfg.n_kv_heads,
                max_seq_len=self.cfg.max_seq_len,
            )
            model = IndicLLM(model_cfg).to(self.device)
        return model

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        dataset = PreferenceDataset(
            data_path=self.cfg.data_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.cfg.max_seq_len,
            max_samples=self.cfg.max_samples,
            seed=self.cfg.seed,
        )
        n_val = max(1, int(len(dataset) * self.cfg.val_fraction))
        n_train = len(dataset) - n_val

        generator = torch.Generator().manual_seed(self.cfg.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False
        )
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, max_batches: int = 20) -> Tuple[float, float, float]:
        """
        Evaluate DPO loss, average reward margin, and preference accuracy.
        """
        self.policy_model.eval()
        total_loss = 0.0
        total_margin = 0.0
        correct_prefs = 0
        total_pairs = 0
        n_batches = 0

        for c_in, c_tgt, r_in, r_tgt in self.val_loader:
            if n_batches >= max_batches:
                break
            c_in, c_tgt = c_in.to(self.device), c_tgt.to(self.device)
            r_in, r_tgt = r_in.to(self.device), r_tgt.to(self.device)

            pi_c_logps = compute_sequence_logps(self.policy_model, c_in, c_tgt)
            pi_r_logps = compute_sequence_logps(self.policy_model, r_in, r_tgt)
            ref_c_logps = compute_sequence_logps(self.ref_model, c_in, c_tgt)
            ref_r_logps = compute_sequence_logps(self.ref_model, r_in, r_tgt)

            loss, c_rewards, r_rewards = dpo_loss(
                pi_c_logps, pi_r_logps, ref_c_logps, ref_r_logps, beta=self.cfg.beta
            )

            total_loss += loss.item()
            margins = c_rewards - r_rewards
            total_margin += margins.sum().item()
            correct_prefs += (margins > 0).sum().item()
            total_pairs += len(margins)
            n_batches += 1

        self.policy_model.train()
        avg_loss = total_loss / max(1, n_batches)
        avg_margin = total_margin / max(1, total_pairs)
        accuracy = correct_prefs / max(1, total_pairs)
        return avg_loss, avg_margin, accuracy

    def train(self) -> None:
        cfg = self.cfg
        log.info("Starting DPO training: %d steps | beta=%.2f | lr=%.2e", cfg.max_steps, cfg.beta, cfg.learning_rate)

        step = 0
        micro_step = 0
        running_loss = 0.0
        t_start = time.time()
        train_iter = iter(self.train_loader)

        while step < cfg.max_steps:
            try:
                c_in, c_tgt, r_in, r_tgt = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                c_in, c_tgt, r_in, r_tgt = next(train_iter)

            c_in, c_tgt = c_in.to(self.device), c_tgt.to(self.device)
            r_in, r_tgt = r_in.to(self.device), r_tgt.to(self.device)

            with torch.no_grad():
                ref_c_logps = compute_sequence_logps(self.ref_model, c_in, c_tgt)
                ref_r_logps = compute_sequence_logps(self.ref_model, r_in, r_tgt)

            pi_c_logps = compute_sequence_logps(self.policy_model, c_in, c_tgt)
            pi_r_logps = compute_sequence_logps(self.policy_model, r_in, r_tgt)

            loss, _, _ = dpo_loss(
                pi_c_logps, pi_r_logps, ref_c_logps, ref_r_logps, beta=cfg.beta
            )
            loss_scaled = loss / cfg.grad_accumulation_steps
            loss_scaled.backward()

            running_loss += loss.item()
            micro_step += 1

            if micro_step % cfg.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1

                if step % cfg.log_every == 0:
                    avg_loss = running_loss / (cfg.log_every * cfg.grad_accumulation_steps)
                    log.info("step %4d/%d | dpo_loss %.4f", step, cfg.max_steps, avg_loss)
                    running_loss = 0.0

                if step % cfg.eval_every == 0 or step == cfg.max_steps:
                    val_loss, val_margin, val_acc = self.evaluate()
                    log.info("  [Validation] step %d: loss=%.4f | margin=%.4f | accuracy=%.1f%%",
                             step, val_loss, val_margin, val_acc * 100.0)

        log.info("DPO alignment completed in %.1f minutes", (time.time() - t_start) / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indic LLM Direct Preference Optimization")
    parser.add_argument("--config", type=str, default=None, help="Path to DPO YAML config")
    parser.add_argument("--base_checkpoint", type=str, default=None, help="Base policy checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Preference JSONL path")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.config:
        cfg = DPOConfig.from_yaml(cli_args.config)
    else:
        cfg = DPOConfig()

    for k, v in vars(cli_args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    trainer = DPOTrainer(cfg)
    trainer.train()

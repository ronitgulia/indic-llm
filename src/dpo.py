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

import json
import logging
import os
import random
from typing import List, Optional, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

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

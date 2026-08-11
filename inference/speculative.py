"""
Indic LLM — Speculative Decoding Engine
========================================
Accelerates autoregressive generation by using a small draft model to propose
candidate tokens that are verified in parallel by the target model in a single
forward pass (Leviathan et al., 2023; Chen et al., 2023).

  - SpeculativeConfig  : Hyperparameters for draft lookahead and sampling
  - SpeculativeMetrics : Throughput, acceptance rate, and latency tracking
  - SpeculativeEngine  : Manages draft proposal, parallel verification, and residual correction
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
sys_path = str(repo_root / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("speculative")


@dataclass
class SpeculativeConfig:
    draft_steps: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: 150
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class SpeculativeMetrics:
    tokens_generated: int
    draft_tokens_proposed: int
    accepted_tokens: int
    target_forward_passes: int
    acceptance_rate: float
    total_time_sec: float
    tokens_per_second: float


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample next token from logits with temperature and top-p filtering."""
    if temperature <= 1e-4:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / max(1e-5, temperature)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        scaled = torch.zeros_like(scaled).scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def speculative_step(
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    draft_token_id: int,
) -> Tuple[bool, Optional[int]]:
    """
    Acceptance test for a single proposed draft token:
      accept_prob = min(1.0, P_target(x) / P_draft(x))
    """
    p_t = p_target[draft_token_id].item()
    p_d = max(1e-8, p_draft[draft_token_id].item())
    alpha = min(1.0, p_t / p_d)

    u = torch.rand(1).item()
    if u <= alpha:
        return True, None

    # Rejection: sample from adjusted residual distribution max(0, P_target - P_draft)
    residual = torch.clamp(p_target - p_draft, min=0.0)
    res_sum = residual.sum().item()
    if res_sum > 1e-8:
        residual_probs = residual / res_sum
        corrected_token = int(torch.multinomial(residual_probs, num_samples=1).item())
    else:
        corrected_token = int(torch.multinomial(p_target, num_samples=1).item())

    return False, corrected_token


class SpeculativeEngine:
    """
    Coordinates speculative decoding between draft and target IndicLLM instances.
    """

    def __init__(
        self,
        target_model: IndicLLM,
        draft_model: IndicLLM,
        tokenizer: spm.SentencePieceProcessor,
        device: Optional[torch.device] = None,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.device = device or next(target_model.parameters()).device

        self.target_model.eval().to(self.device)
        self.draft_model.eval().to(self.device)

    def _get_probs(self, model: IndicLLM, input_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = model(input_ids)
            last_logits = logits[0, -1, :]
            if temperature <= 1e-4:
                probs = torch.zeros_like(last_logits)
                probs[torch.argmax(last_logits)] = 1.0
                return probs
            return F.softmax(last_logits / max(1e-5, temperature), dim=-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[SpeculativeConfig] = None,
    ) -> Tuple[str, SpeculativeMetrics]:
        cfg = config or SpeculativeConfig()
        t_start = time.time()

        prompt_tokens = self.tokenizer.encode(prompt, out_type=int)
        bos_id = self.tokenizer.bos_id()
        if bos_id >= 0 and (not prompt_tokens or prompt_tokens[0] != bos_id):
            prompt_tokens = [bos_id] + prompt_tokens

        eos_id = self.tokenizer.eos_id()
        curr_seq = list(prompt_tokens)

        total_draft_tokens = 0
        accepted_tokens = 0
        target_passes = 0
        generated_token_ids: List[int] = []

        while len(generated_token_ids) < cfg.max_new_tokens:
            draft_tokens: List[int] = []
            draft_probs_list: List[torch.Tensor] = []

            draft_input = torch.tensor([curr_seq], dtype=torch.long, device=self.device)
            for _ in range(cfg.draft_steps):
                p_d = self._get_probs(self.draft_model, draft_input, cfg.temperature)
                tok = sample_token(p_d, temperature=cfg.temperature, top_p=cfg.top_p)
                draft_tokens.append(tok)
                draft_probs_list.append(p_d)
                draft_input = torch.cat(
                    [draft_input, torch.tensor([[tok]], dtype=torch.long, device=self.device)], dim=1
                )
                if tok == eos_id:
                    break

            total_draft_tokens += len(draft_tokens)

            target_input = torch.tensor([curr_seq + draft_tokens], dtype=torch.long, device=self.device)
            target_logits, _ = self.target_model(target_input)
            target_passes += 1

            prefix_len = len(curr_seq)
            all_accepted = True

            for step_idx, draft_tok in enumerate(draft_tokens):
                pos = prefix_len + step_idx - 1
                logits_at_pos = target_logits[0, pos, :]
                p_t = F.softmax(logits_at_pos / max(1e-5, cfg.temperature), dim=-1)
                p_d = draft_probs_list[step_idx]

                accepted, corrected = speculative_step(p_t, p_d, draft_tok)
                if accepted:
                    accepted_tokens += 1
                    curr_seq.append(draft_tok)
                    generated_token_ids.append(draft_tok)
                    if draft_tok == eos_id or len(generated_token_ids) >= cfg.max_new_tokens:
                        break
                else:
                    all_accepted = False
                    if corrected is not None:
                        curr_seq.append(corrected)
                        generated_token_ids.append(corrected)
                    break

            if all_accepted and generated_token_ids and generated_token_ids[-1] != eos_id:
                bonus_logits = target_logits[0, prefix_len + len(draft_tokens) - 1, :]
                bonus_tok = sample_token(bonus_logits, temperature=cfg.temperature, top_p=cfg.top_p)
                curr_seq.append(bonus_tok)
                generated_token_ids.append(bonus_tok)

            if generated_token_ids and generated_token_ids[-1] == eos_id:
                break

        total_time = max(1e-4, time.time() - t_start)
        gen_text = self.tokenizer.decode(generated_token_ids)
        acc_rate = (accepted_tokens / max(1, total_draft_tokens)) * 100.0
        tok_s = len(generated_token_ids) / total_time

        metrics = SpeculativeMetrics(
            tokens_generated=len(generated_token_ids),
            draft_tokens_proposed=total_draft_tokens,
            accepted_tokens=accepted_tokens,
            target_forward_passes=target_passes,
            acceptance_rate=round(acc_rate, 2),
            total_time_sec=round(total_time, 4),
            tokens_per_second=round(tok_s, 2),
        )
        return gen_text, metrics

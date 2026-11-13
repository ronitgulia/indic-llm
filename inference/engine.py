"""
Indic LLM — High-Performance Streaming Inference Engine
=========================================================
Provides low-latency autoregressive generation with:
  - O(1) incremental decoding via KV-cache
  - Real-time token generator streaming
  - Multi-token stop sequence detection
  - Repetition penalty, top-k, and nucleus (top-p) sampling
  - Clean decoupled API for CLI, HTTP server, and Python libraries
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    from model import IndicLLM
except ImportError:
    from src.model import IndicLLM  # type: ignore[no-redef]


@dataclass
class GenerationConfig:
    """Hyperparameters controlling text generation and sampling."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.15
    stop_sequences: List[str] = field(default_factory=list)
    eos_id: Optional[int] = None


class InferenceEngine:
    """
    Manages KV-cached autoregressive text generation for an IndicLLM model.

    Avoids O(T^2) recomputation on each token by pre-filling the prompt once
    and streaming subsequent tokens in O(1) time complexity per step.
    """

    def __init__(
        self,
        model: IndicLLM,
        tokenizer: spm.SentencePieceProcessor,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )
        self.model.eval()
        self.model.to(self.device)

        # Initialize KV-cache for batch size 1
        self.model.init_kv_cache(max_batch_size=1)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[Tuple[str, float]]:
        """
        Generate text token-by-token from a prompt string.

        Yields
        ------
        (token_text, tokens_per_sec) for every emitted token.
        """
        cfg = config or GenerationConfig()
        eos_id = cfg.eos_id if cfg.eos_id is not None else self.tokenizer.eos_id()

        input_ids = self.tokenizer.encode(prompt, out_type=int)
        if not input_ids:
            input_ids = [self.tokenizer.bos_id()]

        # Truncate prompt if longer than model max sequence length
        if len(input_ids) >= self.model.config.max_seq_len:
            input_ids = input_ids[-self.model.config.max_seq_len + 1 :]

        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_ids: List[int] = []
        accumulated_text = ""

        # Reset KV-cache state
        self.model.reset_kv_cache()
        prompt_len = tokens.shape[1]

        # ── Step 1: Prefill prompt in one shot ──────────────────────────────
        logits, _ = self.model(tokens, start_pos=0)
        curr_logits = logits[:, -1, :].clone()

        t_start = time.time()
        tokens_generated = 0

        # ── Step 2: Autoregressive decoding with KV-cache ───────────────────
        for step in range(cfg.max_new_tokens):
            if prompt_len + step >= self.model.config.max_seq_len:
                break

            # Repetition penalty
            if cfg.repetition_penalty != 1.0 and (generated_ids or input_ids):
                seen_tokens = set(generated_ids + input_ids)
                for tid in seen_tokens:
                    if curr_logits[0, tid] > 0:
                        curr_logits[0, tid] /= cfg.repetition_penalty
                    else:
                        curr_logits[0, tid] *= cfg.repetition_penalty

            # Greedy vs sampling decoding
            if cfg.temperature <= 1e-5:
                next_token_id = int(torch.argmax(curr_logits, dim=-1).item())
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            else:
                # Temperature
                scaled_logits = curr_logits / max(cfg.temperature, 1e-8)

                # Top-K filtering
                if cfg.top_k > 0:
                    topk_vals, _ = torch.topk(
                        scaled_logits, min(cfg.top_k, scaled_logits.shape[-1])
                    )
                    scaled_logits[scaled_logits < topk_vals[:, [-1]]] = float("-inf")

                # Top-P (nucleus) filtering
                if cfg.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > cfg.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_tensor = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token_tensor.item()

            if next_token_id == eos_id:
                break

            generated_ids.append(next_token_id)
            tokens_generated += 1

            # Decode single token
            token_str = self.tokenizer.decode([next_token_id])
            accumulated_text += token_str

            # Stop sequence check
            hit_stop = False
            for stop in cfg.stop_sequences:
                if stop and stop in accumulated_text:
                    hit_stop = True
                    break
            if hit_stop:
                break

            elapsed = max(time.time() - t_start, 1e-6)
            tok_per_sec = tokens_generated / elapsed
            yield token_str, tok_per_sec

            # Advance KV-cache by feeding ONLY the newly sampled token
            if step < cfg.max_new_tokens - 1 and prompt_len + step + 1 < self.model.config.max_seq_len:
                logits, _ = self.model(next_token_tensor, start_pos=prompt_len + step)
                curr_logits = logits[:, -1, :].clone()

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Tuple[str, float]:
        """
        Run complete generation and return full generated string + throughput.
        """
        cfg = config or GenerationConfig()
        chunks = []
        final_tok_per_sec = 0.0
        for token_text, tok_s in self.generate_stream(prompt, cfg):
            chunks.append(token_text)
            final_tok_per_sec = tok_s
        full_text = "".join(chunks)

        for stop in cfg.stop_sequences:
            if stop and stop in full_text:
                full_text = full_text.split(stop, 1)[0]

        return full_text, final_tok_per_sec

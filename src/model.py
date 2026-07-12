"""
Indic LLM - Transformer Model Architecture
==========================================
Llama-style decoder-only transformer with:
  - RoPE (Rotary Positional Embeddings) with optional NTK-aware scaling
  - RMSNorm instead of LayerNorm
  - SwiGLU activation in FFN
  - Grouped Query Attention (GQA) with optional Flash Attention 2 backend
  - KV-cache for O(1) incremental decoding
  - Depth-scaled weight initialisation (GPT-2 style)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Flash Attention — gracefully fall back to vanilla SDPA
try:
    from flash_attn import flash_attn_func  # type: ignore[import]
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size: int = 32000         # SentencePiece vocab
    dim: int = 512                  # embedding dimension
    n_layers: int = 8               # transformer layers
    n_heads: int = 8                # attention heads (Q)
    n_kv_heads: int = 4             # GQA key-value heads
    ffn_dim_multiplier: float = 2.67  # FFN hidden = dim * multiplier
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0
    pad_id: int = 0

    # Flash Attention 2 (requires the flash-attn package)
    use_flash_attn: bool = False

    # RoPE NTK-aware scaling for long-context generalisation
    # Set rope_scaling_factor > 1.0 to extend effective context length
    rope_scaling_factor: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        # Round to nearest multiple of 256 for hardware efficiency
        raw = int(self.dim * self.ffn_dim_multiplier)
        return (raw + 255) // 256 * 256


# ─────────────────────────────────────────
#  RMSNorm
# ─────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — faster than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ─────────────────────────────────────────
#  Rotary Positional Embeddings (RoPE)
# ─────────────────────────────────────────

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Pre-compute complex exponentials for RoPE.

    Parameters
    ----------
    dim            : Head dimension.
    end            : Maximum sequence length to pre-compute.
    theta          : RoPE base (default: 10 000).
    scaling_factor : NTK-aware scaling factor. Values > 1.0 extend
                     the effective context window without fine-tuning.
    """
    if scaling_factor != 1.0:
        # NTK-aware interpolation: scale theta to shift frequencies
        theta = theta * scaling_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    # freqs_cis: (T, head_dim/2) → broadcast over batch and heads
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ─────────────────────────────────────────
#  KV Cache
# ─────────────────────────────────────────

class KVCache(nn.Module):
    """
    Pre-allocated key-value cache for autoregressive inference.

    Avoids recomputing past KV pairs at every generation step,
    reducing inference complexity from O(T²) to O(T).

    Parameters
    ----------
    max_batch_size : Maximum batch size to pre-allocate for.
    max_seq_len    : Maximum sequence length to pre-allocate for.
    n_kv_heads     : Number of key-value heads.
    head_dim       : Dimension per head.
    device         : Target device.
    dtype          : Storage dtype (default: float16 for memory efficiency).
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.seq_len: int = 0

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new KV pairs to the cache and return the full accumulated KV.

        Parameters
        ----------
        k         : New keys   (B, T_new, n_kv_heads, head_dim).
        v         : New values (B, T_new, n_kv_heads, head_dim).
        start_pos : Token position where the new sequence starts.

        Returns
        -------
        (k_full, v_full) accumulated up to start_pos + T_new.
        """
        T_new = k.shape[1]
        self.k_cache[:k.shape[0], start_pos: start_pos + T_new] = k
        self.v_cache[:v.shape[0], start_pos: start_pos + T_new] = v
        self.seq_len = start_pos + T_new

        k_full = self.k_cache[:k.shape[0], :self.seq_len]
        v_full = self.v_cache[:v.shape[0], :self.seq_len]
        return k_full, v_full

    def reset(self) -> None:
        """Clear the cache (call between independent generation requests)."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


# ─────────────────────────────────────────
#  Grouped Query Attention (GQA)
# ─────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with GQA:
    n_heads query heads share n_kv_heads key/value heads.

    Variants supported:
      n_kv_heads == n_heads  → standard MHA
      n_kv_heads == 1        → Multi-Query Attention (MQA)
      1 < n_kv_heads < heads → GQA (LLaMA-2 / Mistral style)

    Optionally uses Flash Attention 2 when available and configured.
    KV-cache is supported for efficient autoregressive decoding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # repeat factor for KV
        self.use_flash = config.use_flash_attn and _FLASH_AVAILABLE

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # KV-cache (populated via init_kv_cache before generation)
        self.kv_cache: Optional[KVCache] = None

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Allocate a KV-cache for this attention layer."""
        self.kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads (GQA expansion)."""
        if self.n_rep == 1:
            return x
        B, T, n_kv, hd = x.shape
        return (
            x[:, :, :, None, :]
            .expand(B, T, n_kv, self.n_rep, hd)
            .reshape(B, T, n_kv * self.n_rep, hd)
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV-cache update during inference
        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(xk, xv, start_pos)

        # Expand KV heads for GQA
        xk = self._repeat_kv(xk)
        xv = self._repeat_kv(xv)

        if self.use_flash:
            # Flash Attention expects (B, T, heads, head_dim) in bfloat16/float16
            xq_fa = xq.to(torch.bfloat16)
            xk_fa = xk.to(torch.bfloat16)
            xv_fa = xv.to(torch.bfloat16)
            out = flash_attn_func(xq_fa, xk_fa, xv_fa, causal=True)
            out = out.to(x.dtype).reshape(B, T, -1)
        else:
            # Scaled dot-product attention — (B, heads, T, head_dim)
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / scale

            if mask is not None:
                scores = scores + mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.dropout(scores)

            out = torch.matmul(scores, xv)          # (B, heads, T, head_dim)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ─────────────────────────────────────────
#  SwiGLU Feed-Forward Network
# ─────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN — used in PaLM, LLaMA:
      FFN(x) = SiLU(xW1) ⊙ (xW3) · W2
    Three weight matrices (gate, up, down) with no bias.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hd = config.ffn_hidden_dim
        self.w1 = nn.Linear(config.dim, hd, bias=False)   # gate
        self.w2 = nn.Linear(hd, config.dim, bias=False)   # down
        self.w3 = nn.Linear(config.dim, hd, bias=False)   # up
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────
#  Transformer Block
# ─────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────
#  Indic LLM — Main Model
# ─────────────────────────────────────────

class IndicLLM(nn.Module):
    """
    Decoder-only causal language model for Indic languages.

    Architecture: LLaMA-2 style transformer
      - RMSNorm (pre-norm)
      - RoPE positional embeddings (with optional NTK scaling)
      - Grouped Query Attention (GQA)
      - SwiGLU activation (FFN)
      - Weight tying between token embeddings and output projection

    Supports:
      - Flash Attention 2 (requires flash-attn package)
      - KV-cache for O(1)-per-step autoregressive decoding
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=config.pad_id
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Pre-compute RoPE frequencies once
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta,
                config.rope_scaling_factor,
            ),
        )

        # BUG FIX: initialise weights BEFORE tying so that _init_weights()
        # only sees distinct tensors.  Previously the tie was set first, causing
        # the shared tensor to be initialised twice (once as Linear with the
        # depth-scaled std, then again as Embedding with std=0.02), silently
        # discarding the depth-scaled init for the output projection.
        self._init_weights()

        # Weight tying: embedding and output share weights (saves ~20M params).
        # Done AFTER _init_weights() so the embedding initialisation wins and
        # the output projection pointer is updated cleanly.
        self.output.weight = self.tok_embeddings.weight

    def _init_weights(self) -> None:
        """
        Depth-scaled weight initialisation (GPT-2 style).
        Output projections (wo, w2) are scaled by 1/√(2 * n_layers)
        to compensate for residual accumulation depth.
        """
        residual_scale = (2 * self.config.n_layers) ** -0.5
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                # Scale residual projections
                if name.endswith((".wo", ".w2")):
                    std *= residual_scale
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper triangular causal mask — prevents attending to future tokens."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    # ------------------------------------------------------------------
    # KV-cache management
    # ------------------------------------------------------------------

    def init_kv_cache(
        self,
        max_batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Allocate KV-caches for all attention layers.
        Call once before starting a generation loop; reset between requests.

        Parameters
        ----------
        max_batch_size : Maximum generation batch size.
        dtype          : Storage dtype (float16 or bfloat16 recommended).
        """
        device = next(self.parameters()).device
        for layer in self.layers:
            layer.attn.init_kv_cache(
                max_batch_size=max_batch_size,
                max_seq_len=self.config.max_seq_len,
                device=device,
                dtype=dtype,
            )

    def reset_kv_cache(self) -> None:
        """Clear all KV-caches between generation requests."""
        for layer in self.layers:
            if layer.attn.kv_cache is not None:
                layer.attn.kv_cache.reset()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ):
        """
        Parameters
        ----------
        tokens    : (B, T) input token ids.
        targets   : (B, T) target token ids for loss computation (optional).
        start_pos : Starting position for KV-cache (0 during training).

        Returns
        -------
        logits : (B, T, vocab_size)
        loss   : scalar cross-entropy loss if ``targets`` is provided, else None.
        """
        B, T = tokens.shape
        device = tokens.device

        x = self.tok_embeddings(tokens)           # (B, T, dim)
        freqs_cis = self.freqs_cis[start_pos: start_pos + T]

        # During cached inference, only the new token needs a mask of shape (1, 1)
        if T == 1:
            mask = None
        else:
            mask = self._make_causal_mask(T, device)

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, start_pos)

        x = self.norm(x)
        logits = self.output(x).float()           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_id,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        eos_id: Optional[int] = None,
        use_kv_cache: bool = True,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature + top-p + top-k sampling.

        When ``use_kv_cache=True`` the first forward pass processes the full
        prompt and caches past KV pairs; subsequent steps process one token
        at a time in O(1) per step.

        Parameters
        ----------
        tokens         : (1, T) prompt token ids.
        max_new_tokens : Maximum number of tokens to generate.
        temperature    : Sampling temperature (lower → more deterministic).
        top_p          : Nucleus sampling cumulative probability threshold.
        top_k          : Number of top candidates for top-k filtering.
        eos_id         : If set, stop generation when this token is sampled.
        use_kv_cache   : Use KV-cache for fast incremental decoding.

        Returns
        -------
        (1, T + generated) full token sequence including the prompt.
        """
        self.eval()

        if use_kv_cache:
            self.init_kv_cache(max_batch_size=tokens.shape[0])
            self.reset_kv_cache()

        prompt_len = tokens.shape[1]

        # Prefill: process the prompt in one shot
        if use_kv_cache:
            logits, _ = self(tokens, start_pos=0)
            start_pos = prompt_len
        else:
            start_pos = 0

        for i in range(max_new_tokens):
            if use_kv_cache:
                # Decode: feed only the last token
                ctx = tokens[:, -1:]
                logits, _ = self(ctx, start_pos=start_pos + i)
            else:
                ctx = tokens if tokens.shape[1] <= self.config.max_seq_len \
                      else tokens[:, -self.config.max_seq_len:]
                logits, _ = self(ctx, start_pos=0)

            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(tokens.shape[0]):
                    for token_id in set(tokens[b].tolist()):
                        if next_logits[b, token_id] < 0:
                            next_logits[b, token_id] *= repetition_penalty
                        else:
                            next_logits[b, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return tokens

    # ------------------------------------------------------------------
    # Introspection utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def model_summary(self) -> str:
        """
        Return a multi-line human-readable summary of the model architecture,
        including per-component parameter counts.
        """
        rows: List[str] = []
        rows.append("=" * 56)
        rows.append(f"  IndicLLM  —  {self.num_parameters(trainable_only=False) / 1e6:.2f}M params")
        rows.append("=" * 56)
        rows.append(f"  {'Component':<30} {'Params':>10}")
        rows.append("-" * 56)

        def _fmt(n: int) -> str:
            return f"{n / 1e6:.3f}M" if n >= 1_000_000 else f"{n:,}"

        rows.append(
            f"  {'tok_embeddings':<30} {_fmt(self.tok_embeddings.weight.numel()):>10}"
        )
        for i, layer in enumerate(self.layers):
            n = sum(p.numel() for p in layer.parameters())
            rows.append(f"  {'layer.' + str(i):<30} {_fmt(n):>10}")

        rows.append(f"  {'norm':<30} {_fmt(self.norm.weight.numel()):>10}")
        rows.append("=" * 56)
        rows.append(f"  Vocab      : {self.config.vocab_size:,}")
        rows.append(f"  Dim        : {self.config.dim}")
        rows.append(f"  Layers     : {self.config.n_layers}")
        rows.append(f"  Heads      : {self.config.n_heads}Q / {self.config.n_kv_heads}KV")
        rows.append(f"  FFN hidden : {self.config.ffn_hidden_dim}")
        rows.append(f"  Max seq    : {self.config.max_seq_len}")
        rows.append(
            f"  Flash Attn : {'yes (flash-attn)' if _FLASH_AVAILABLE and self.config.use_flash_attn else 'no (vanilla SDPA)'}"
        )
        rows.append(f"  RoPE scale : {self.config.rope_scaling_factor}x")
        rows.append("=" * 56)
        return "\n".join(rows)


# ─────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────

if __name__ == "__main__":
    config = ModelConfig()
    model = IndicLLM(config)

    print(model.model_summary())

    # Forward pass test
    dummy_tokens = torch.randint(0, config.vocab_size, (2, 64))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(dummy_tokens, dummy_targets)
    print(f"\n  Forward pass: logits {logits.shape}, loss = {loss.item():.4f}")

    # KV-cache generation test
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.9, use_kv_cache=False)
    print(f"  Generation  : {prompt.shape[1]} → {generated.shape[1]} tokens")
    print("\n  Model ready!")

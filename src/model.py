"""
Indic LLM - Transformer Model Architecture
==========================================
Llama-style decoder-only transformer with:
  - RoPE (Rotary Positional Embeddings)
  - RMSNorm instead of LayerNorm
  - SwiGLU activation in FFN
  - Grouped Query Attention (GQA) support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size: int = 32000         # SentencePiece vocab
    dim: int = 512                  # embedding dimension
    n_layers: int = 8               # transformer layers
    n_heads: int = 8                # attention heads
    n_kv_heads: int = 4            # GQA key-value heads
    ffn_dim_multiplier: float = 2.67  # FFN hidden = dim * multiplier
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0
    pad_id: int = 0

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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Pre-compute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    freqs_cis = freqs_cis[:xq.shape[1], :].unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ─────────────────────────────────────────
#  Grouped Query Attention (GQA)
# ─────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with GQA:
    n_heads query heads share n_kv_heads key/value heads.
    When n_kv_heads == n_heads → standard MHA.
    When n_kv_heads == 1       → MQA.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # repeat factor for KV

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of query heads."""
        if self.n_rep == 1:
            return x
        B, T, n_kv, hd = x.shape
        return x[:, :, :, None, :].expand(B, T, n_kv, self.n_rep, hd)\
                                   .reshape(B, T, n_kv * self.n_rep, hd)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat KV for GQA
        xk = self._repeat_kv(xk)
        xv = self._repeat_kv(xv)

        # Scaled dot-product attention  (B, heads, T, head_dim)
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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + residual
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────
#  Indic LLM — Main Model
# ─────────────────────────────────────────

class IndicLLM(nn.Module):
    """
    Decoder-only causal language model for Indic languages.

    Architecture: LLaMA-style transformer
      - RMSNorm normalization
      - RoPE positional embeddings
      - Grouped Query Attention
      - SwiGLU activation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim,
                                           padding_idx=config.pad_id)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: embedding and output share weights
        self.output.weight = self.tok_embeddings.weight

        # Pre-compute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_seq_len, config.rope_theta),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper triangular causal mask to prevent attending to future tokens."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, tokens: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        """
        Args:
            tokens:  (B, T) input token ids
            targets: (B, T) target token ids for computing loss (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss if targets provided, else None
        """
        B, T = tokens.shape
        device = tokens.device

        x = self.tok_embeddings(tokens)           # (B, T, dim)
        freqs_cis = self.freqs_cis[:T]
        mask = self._make_causal_mask(T, device)

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        logits = self.output(x).float()           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Shift: predict token at position i from tokens 0..i-1
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_p: float = 0.95,
                 top_k: int = 50) -> torch.Tensor:
        """
        Autoregressive generation with temperature + top-p + top-k sampling.

        Args:
            tokens:         (1, T) prompt token ids
            max_new_tokens: max tokens to generate
            temperature:    sampling temperature (lower = more deterministic)
            top_p:          nucleus sampling threshold
            top_k:          top-k candidates to sample from

        Returns:
            (1, T + max_new_tokens) generated token ids
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len
            ctx = tokens if tokens.shape[1] <= self.config.max_seq_len \
                  else tokens[:, -self.config.max_seq_len:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────

if __name__ == "__main__":
    config = ModelConfig()
    model = IndicLLM(config)

    total_params = model.num_parameters(trainable_only=False)
    print(f"\n IndicLLM Model Summary")
    print(f"  Layers      : {config.n_layers}")
    print(f"  Dim         : {config.dim}")
    print(f"  Heads       : {config.n_heads} (Q) / {config.n_kv_heads} (KV)")
    print(f"  FFN hidden  : {config.ffn_hidden_dim}")
    print(f"  Vocab size  : {config.vocab_size}")
    print(f"  Max seq len : {config.max_seq_len}")
    print(f"  Parameters  : {total_params / 1e6:.2f}M")

    # Forward pass test
    dummy_tokens = torch.randint(0, config.vocab_size, (2, 64))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(dummy_tokens, dummy_targets)
    print(f"\n  Forward pass: logits {logits.shape}, loss = {loss.item():.4f}")

    # Generation test
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.9)
    print(f"  Generation  : {prompt.shape[1]} → {generated.shape[1]} tokens")
    print("\n Model ready!")

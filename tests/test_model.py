"""
Unit tests for IndicLLM architecture components:
  - ModelConfig validation and parameter sizing
  - Forward pass shapes and loss calculation
  - Grouped Query Attention (GQA) & Rotary Positional Embeddings (RoPE)
  - SwiGLU FFN dimensions
  - RMSNorm normalization properties
  - Sliding Window Attention (SWA) causal mask
  - KV-cache decoding equivalence (incremental O(1) vs full O(T^2))
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (
    IndicLLM,
    ModelConfig,
    RMSNorm,
    SwiGLUFFN,
    apply_rotary_emb,
    precompute_freqs_cis,
)


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        dropout=0.0,
        pad_id=0,
    )


def test_model_instantiation(tiny_config: ModelConfig):
    model = IndicLLM(tiny_config)
    n_params = model.num_parameters()
    assert n_params > 0
    # Embedding weight and output head should be tied
    assert model.tok_embeddings.weight is model.output.weight


def test_forward_pass_without_targets(tiny_config: ModelConfig):
    model = IndicLLM(tiny_config)
    model.eval()

    batch_size, seq_len = 2, 16
    tokens = torch.randint(1, tiny_config.vocab_size, (batch_size, seq_len))

    logits, loss = model(tokens)
    assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)
    assert loss is None


def test_forward_pass_with_targets(tiny_config: ModelConfig):
    model = IndicLLM(tiny_config)
    model.train()

    batch_size, seq_len = 2, 16
    tokens = torch.randint(1, tiny_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(1, tiny_config.vocab_size, (batch_size, seq_len))

    logits, loss = model(tokens, targets=targets)
    assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_forward_pass_with_custom_ignore_index(tiny_config: ModelConfig):
    model = IndicLLM(tiny_config)
    model.eval()

    tokens = torch.randint(1, tiny_config.vocab_size, (1, 8))
    # All targets set to -100 except one token
    targets = torch.full((1, 8), -100, dtype=torch.long)
    targets[0, 3] = 42

    logits, loss = model(tokens, targets=targets, ignore_index=-100)
    assert loss is not None
    assert loss.item() > 0.0


def test_rope_frequencies_and_rotation(tiny_config: ModelConfig):
    dim = 16
    seq_len = 32
    freqs_cis = precompute_freqs_cis(dim, seq_len, theta=10000.0)

    assert freqs_cis.shape == (seq_len, dim // 2)
    assert freqs_cis.dtype == torch.complex64

    # Test apply_rotary_emb
    xq = torch.randn(2, seq_len, 4, dim)
    xk = torch.randn(2, seq_len, 2, dim)
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    assert xq_rot.shape == xq.shape
    assert xk_rot.shape == xk.shape
    # Norm along head_dim is approximately preserved under unitary rotation
    assert torch.allclose(xq.norm(dim=-1), xq_rot.norm(dim=-1), atol=1e-4)


def test_swiglu_dimension_rounding():
    cfg = ModelConfig(dim=128, ffn_dim_multiplier=2.67)
    # 128 * 2.67 = 341.76 -> rounded up to next multiple of 256 is 512
    assert cfg.ffn_hidden_dim % 256 == 0
    assert cfg.ffn_hidden_dim >= int(128 * 2.67)

    ffn = SwiGLUFFN(cfg)
    x = torch.randn(2, 8, cfg.dim)
    out = ffn(x)
    assert out.shape == x.shape


def test_rmsnorm():
    dim = 32
    norm = RMSNorm(dim)
    x = torch.randn(2, 4, dim) * 10.0 + 5.0
    out = norm(x)

    assert out.shape == x.shape
    # Root mean square should be close to 1 before weight scale
    rms = torch.sqrt(out.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_sliding_window_mask():
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=64, sliding_window=4)
    model = IndicLLM(cfg)

    mask = model._make_causal_mask(seq_len=8, device=torch.device("cpu"))
    # Position (0, 7) cannot attend to position 0 because 7 - 0 >= window_size (4)
    assert mask[0, 0, 7, 0] == float("-inf")
    # Position (0, 7) CAN attend to position 4 because 7 - 4 = 3 < window_size
    assert mask[0, 0, 7, 4] == 0.0
    # Future positions are masked (-inf)
    assert mask[0, 0, 2, 5] == float("-inf")


def test_kv_cache_equivalence(tiny_config: ModelConfig):
    """
    Verify that incremental decoding using the KV-cache yields the EXACT same
    logits as full-sequence non-cached forward evaluation.
    """
    torch.manual_seed(42)
    model = IndicLLM(tiny_config)
    model.eval()

    prompt_len = 5
    gen_len = 4

    prompt = torch.randint(1, tiny_config.vocab_size, (1, prompt_len))
    tokens = prompt.clone()

    # 1. Autoregressive generation with KV-cache
    model.init_kv_cache(max_batch_size=1)
    model.reset_kv_cache()

    # Prefill prompt
    logits_prefill, _ = model(prompt, start_pos=0)
    cached_logits = [logits_prefill[:, -1, :]]

    # Step-by-step decode
    for i in range(gen_len):
        next_tok = torch.randint(1, tiny_config.vocab_size, (1, 1))
        tokens = torch.cat([tokens, next_tok], dim=1)
        logits_step, _ = model(next_tok, start_pos=prompt_len + i)
        cached_logits.append(logits_step[:, -1, :])

    # 2. Non-cached full-sequence forward pass
    logits_full, _ = model(tokens, start_pos=0)

    # Check prompt last token logit
    assert torch.allclose(cached_logits[0], logits_full[:, prompt_len - 1, :], atol=1e-4)

    # Check each subsequent generated token logit
    for step_idx in range(gen_len):
        step_cached = cached_logits[step_idx + 1]
        step_full = logits_full[:, prompt_len + step_idx, :]
        assert torch.allclose(step_cached, step_full, atol=1e-4), (
            f"KV-cache mismatch at step {step_idx}: "
            f"max diff = {(step_cached - step_full).abs().max().item()}"
        )

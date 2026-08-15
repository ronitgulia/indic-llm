# Indic LLM — Speculative Decoding & Assisted Generation Guide

## 1. Introduction
Speculative decoding reduces the per-token decoding latency of autoregressive transformers by drafting candidate sequences using a lightweight draft model ($M_{draft}$) and verifying all candidates simultaneously with the target model ($M_{target}$) in a single forward pass.

## 2. Mathematical Formulation
Given prompt sequence $x_{1:t}$:
1. **Drafting:** Draft model produces $K$ lookahead tokens.
2. **Parallel Verification:** Target model evaluates candidates in one batched forward pass.
3. **Speculative Acceptance:**
   $$\alpha_i = \min\left(1.0, \frac{P_{target}(y_i)}{P_{draft}(y_i)}\right)$$
4. **Residual Correction:** If rejected, resample from adjusted distribution.

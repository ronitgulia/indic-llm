# Indic LLM

![CI](https://github.com/ronitgulia/indic-llm/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

A research-grade, end-to-end training framework for **decoder-only language models** focused on Indic languages (Hindi-first, multilingual-ready). Built on a LLaMA-2-style architecture with modern training best-practices.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                        IndicLLM (decoder-only)                    │
│                                                                   │
│  Token Embeddings  →  N × TransformerBlock  →  RMSNorm  →  LM Head │
│                                                                   │
│  TransformerBlock                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  RMSNorm → GroupedQueryAttention (GQA + RoPE) → Residual   │  │
│  │  RMSNorm → SwiGLU FFN                         → Residual   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Key design choices:                                              │
│    • Pre-norm residuals (LLaMA-2 style)                           │
│    • RoPE positional encoding (with optional NTK scaling)         │
│    • GQA: N query heads share M key-value heads (M ≤ N)           │
│    • SwiGLU activation: FFN(x) = SiLU(xW₁) ⊙ (xW₃) · W₂         │
│    • Weight tying (embedding ↔ LM head)                           │
│    • Depth-scaled weight initialisation (GPT-2 style)             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
indic-llm/
├── src/
│   ├── model.py          # Transformer architecture (GQA, RoPE, SwiGLU, KV-cache)
│   ├── train.py          # Training loop (AdamW, cosine LR, AMP, W&B, val loss)
│   ├── tokenizer.py      # IndicTokenizer class wrapping SentencePiece BPE
│   └── preprocess.py     # Text cleaning pipeline (dedup, quality filter, schemas)
├── data/
│   ├── download_datasets.py   # Download corpora from HuggingFace
│   └── processed/             # Cleaned JSONL files (git-ignored)
├── eval/
│   ├── evaluate.py            # Perplexity + generation quality evaluation harness
│   └── human_eval/
│       └── sample_prompts.jsonl   # 20 Hindi evaluation prompts
├── inference/
│   └── chat.py                # Interactive chat CLI + batch inference
├── configs/
│   └── train_config.yaml      # All hyperparameters with scaling guide
├── .github/workflows/
│   └── ci.yml                 # Lint (ruff) + type-check (mypy) + smoke test
├── requirements.txt
└── CONTRIBUTING.md
```

---

## Quickstart

### 1 — Environment setup

```bash
git clone https://github.com/ronitgulia/indic-llm.git
cd indic-llm
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Download datasets

Downloads IITB Hindi–English parallel corpus, Sangraha Hindi, and Indic-Instruct from HuggingFace.

```bash
python data/download_datasets.py
```

### 3 — Preprocess corpora

```bash
# Sangraha (monolingual)
python src/preprocess.py --schema sangraha \
    --input data/raw/sangraha_hi.jsonl \
    --output data/processed/sangraha_clean.jsonl

# Hindi–English parallel
python src/preprocess.py --schema parallel \
    --input data/raw/hindi_english_parallel.jsonl \
    --output data/processed/hindi_parallel_clean.jsonl

# Indic-Instruct
python src/preprocess.py --schema instruct \
    --input data/raw/indic_instruct_hi.jsonl \
    --output data/processed/indic_instruct_clean.jsonl
```

### 4 — Train tokenizer

```bash
python src/tokenizer.py train \
    --data_dir data/processed \
    --output data/tokenizer/indic_spm \
    --vocab_size 32000

# Verify round-trip
python src/tokenizer.py test --model data/tokenizer/indic_spm.model
```

### 5 — Train model

**With defaults (reads `configs/train_config.yaml`):**
```bash
python src/train.py --config configs/train_config.yaml
```

**Override specific flags:**
```bash
python src/train.py \
    --config configs/train_config.yaml \
    --batch_size 16 \
    --max_steps 20000 \
    --use_wandb \
    --experiment_name hindi-lm-v1
```

**Resume from checkpoint:**
```bash
python src/train.py --resume_from checkpoints/step_005000.pt
```

### 6 — Evaluate

```bash
python eval/evaluate.py \
    --checkpoint checkpoints/step_010000.pt \
    --data data/processed/sangraha_clean.jsonl \
    --tokenizer data/tokenizer/indic_spm.model \
    --prompts eval/human_eval/sample_prompts.jsonl \
    --output eval/results/report.json
```

### 7 — Chat / Inference

```bash
# Interactive chat (Hindi + English):
python inference/chat.py --checkpoint checkpoints/step_010000.pt

# Single prompt:
python inference/chat.py \
    --checkpoint checkpoints/step_010000.pt \
    --prompt "भारत के बारे में बताओ"

# Batch inference:
python inference/chat.py \
    --checkpoint checkpoints/step_010000.pt \
    --batch_file prompts.txt \
    --output_file results/output.jsonl
```

---

## Model Variants

| Variant | `dim` | `layers` | `heads` (Q/KV) | Params  |
|---------|-------|----------|----------------|---------|
| Small   | 512   | 8        | 8 / 4          | ~25 M   |
| Medium  | 768   | 12       | 12 / 6         | ~120 M  |
| Large   | 1024  | 24       | 16 / 8         | ~350 M  |

Edit `configs/train_config.yaml` to switch variants — no code changes required.

---

## Key Features

| Feature | Details |
|---------|---------|
| **Architecture** | LLaMA-2 style decoder-only transformer |
| **Attention** | Grouped Query Attention (GQA) — configurable Q/KV head ratio |
| **Positional Encoding** | RoPE with optional NTK-aware long-context scaling |
| **FFN** | SwiGLU — `FFN(x) = SiLU(xW₁) ⊙ (xW₃) · W₂` |
| **Normalisation** | RMSNorm (pre-norm residuals) |
| **Inference** | KV-cache for O(1)-per-step autoregressive decoding |
| **Flash Attention** | Optional Flash Attention 2 backend (requires `flash-attn`) |
| **Training** | AdamW, cosine LR with warmup, gradient accumulation, AMP |
| **Monitoring** | Weights & Biases integration (optional, `--use_wandb`) |
| **Checkpointing** | Best-N checkpoint management with metadata JSON |
| **Evaluation** | Perplexity + distinct-1/2 + repetition rate metrics |
| **Tokenizer** | 32K BPE SentencePiece, Devanagari + Latin coverage |

---

## Training Configuration

The YAML config in `configs/train_config.yaml` controls all hyperparameters:

```yaml
model:
  dim: 512
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4          # GQA — Q heads : KV heads = 2 : 1
  max_seq_len: 512
  dropout: 0.1

training:
  batch_size: 8
  grad_accumulation_steps: 4   # effective batch = 32
  learning_rate: 3.0e-4
  min_lr: 3.0e-5               # cosine decay floor
  warmup_steps: 200
  max_steps: 10000
  weight_decay: 0.1
  grad_clip: 1.0
```

---

## Requirements

```
torch >= 2.0
sentencepiece >= 0.1.99
datasets >= 2.14         # for download_datasets.py
pyyaml >= 6.0            # for YAML config loading
wandb >= 0.16            # optional — Weights & Biases logging
flash-attn >= 2.5        # optional — GPU only
```

Install everything:
```bash
pip install -r requirements.txt
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, coding standards, branch strategy, and PR checklist.

---

## License

MIT — see [LICENSE](LICENSE).

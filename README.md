# Indic LLM

![CI](https://github.com/ronitgulia/indic-llm/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

A research-grade, end-to-end training framework for **decoder-only language models** focused on Indic languages (starting with Hindi, but built to be multilingual). I put this together using a LLaMA-2-style architecture and modern training best practices to make it easy to experiment and train models from scratch!

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
│   ├── lora.py           # Low-Rank Adaptation (LoRA) PEFT & adapter merging
│   ├── rag.py            # Indic RAG (chunker, BM25, dense, RRF hybrid retriever)
│   ├── train.py          # Pre-training loop (AdamW, cosine LR, AMP, W&B, val loss)
│   ├── sft.py            # Supervised fine-tuning with LoRA & prompt loss masking
│   ├── dpo.py            # Direct Preference Optimization (DPO) alignment
│   ├── export.py         # HuggingFace & dynamic INT8 export bridge
│   ├── tokenizer.py      # IndicTokenizer class wrapping SentencePiece BPE
│   └── preprocess.py     # Text cleaning pipeline (dedup, quality filter, schemas)
├── data/
│   ├── download_datasets.py   # Download corpora from HuggingFace
│   └── processed/             # Cleaned JSONL files (git-ignored)
├── eval/
│   ├── benchmark_suite.py     # Multi-choice log-likelihood evaluation & leaderboard
│   ├── evaluate.py            # Perplexity + generation quality evaluation harness
│   ├── benchmarks.py          # BLEU, ChrF, token accuracy standalone benchmarks
│   ├── tasks/
│   │   └── indic_qa_samples.jsonl # Curated Indic multiple-choice QA benchmarks
│   └── human_eval/
│       └── sample_prompts.jsonl   # 20 Hindi evaluation prompts
├── inference/
│   ├── engine.py              # KV-cached streaming autoregressive generation engine
│   ├── chat.py                # Interactive streaming chat CLI + batch inference
│   └── serve.py               # OpenAI-compatible streaming HTTP API server
├── configs/
│   ├── train_config.yaml      # Pre-training hyperparameters with scaling guide
│   ├── sft_config.yaml        # Instruction fine-tuning hyperparameters
│   ├── lora_sft_config.yaml   # Parameter-efficient LoRA SFT hyperparameters
│   ├── dpo_config.yaml        # Preference alignment hyperparameters
│   ├── eval_config.yaml       # Multi-task benchmark evaluation settings
│   └── rag_config.yaml        # Indic RAG retrieval & generation settings
├── tests/                     # Comprehensive pytest test suite (58 unit tests)
├── .github/workflows/
│   └── ci.yml                 # Lint (ruff) + type-check + pytest suite
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

### 5b — Supervised Fine-Tuning (SFT)

Fine-tune a pre-trained base checkpoint on instruction datasets with target-only loss masking:

```bash
python src/sft.py \
    --config configs/sft_config.yaml \
    --base_checkpoint checkpoints/step_010000.pt
```

### 5c — Direct Preference Optimization (DPO)

Align model outputs with human feedback using paired preferences:

```bash
python src/dpo.py \
    --config configs/dpo_config.yaml \
    --base_checkpoint checkpoints/sft/sft_best.pt
```

### 5d — Parameter-Efficient Fine-Tuning (LoRA)

Train lightweight low-rank adapters (< 1% trainable parameters, 4x lower memory footprint):

```bash
python src/sft.py \
    --config configs/lora_sft_config.yaml \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16.0
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

### 6b — Indic Benchmark Suite & Leaderboard

Evaluate multi-choice question answering accuracy using length-normalized sequence log-likelihood:

```bash
python eval/benchmark_suite.py \
    --checkpoint checkpoints/step_010000.pt \
    --tokenizer data/tokenizer/indic_spm.model \
    --task eval/tasks/indic_qa_samples.jsonl \
    --output eval/results/benchmark_results.json \
    --report_md eval/results/benchmark_report.md
```

### 7 — Chat / Inference

```bash
# Interactive streaming chat (Hindi + English):
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

### 7b — HTTP API Serving

Launch an OpenAI-compatible REST API server supporting SSE token streaming:

```bash
python inference/serve.py \
    --checkpoint checkpoints/step_010000.pt \
    --port 8000
```

### 7c — Retrieval-Augmented Generation (RAG)

Ground answers in external Indic text corpora with sentence-aware chunking and citation attribution:

```bash
python src/rag.py \
    --checkpoint checkpoints/step_010000.pt \
    --tokenizer data/tokenizer/indic_spm.model \
    --corpus data/knowledge.txt \
    --query "भारत की राजधानी क्या है?" \
    --retriever hybrid \
    --top_k 3
```

### 8 — Export & Quantization

Export checkpoints for Hugging Face Transformers or dynamic INT8 CPU inference:

```bash
# Export to Hugging Face format:
python src/export.py --checkpoint checkpoints/step_010000.pt --format hf --output_dir export/hf_model

# Quantize to dynamic INT8 for lightweight edge deployment:
python src/export.py --checkpoint checkpoints/step_010000.pt --format int8 --output_path checkpoints/indic_int8.pt
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
| **PEFT (LoRA)** | Low-Rank Adaptation for attention/MLP layers (< 1% trainable params) with weight merging |
| **RAG** | Sentence-aware Devanagari chunking (।, ॥), BM25 + Dense RRF hybrid search, citation attribution |
| **Evaluation** | Perplexity + token accuracy + BLEU/ChrF + Multi-Choice QA Benchmark Suite with Leaderboards |
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


### 6. Synthetic Instruction Evolution (Evol-Instruct)
Synthesize culturally relevant, multi-step instruction-response pairs for Indic languages:
```bash
python data/generate_synthetic.py --count 50 --languages hi bn ta te --output data/synthetic_samples.jsonl
```

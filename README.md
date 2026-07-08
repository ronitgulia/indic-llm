# Indic LLM

A repository for training, evaluating, and running inference on Large Language Models for Indic languages (Hindi-first, with multilingual support).

## Pipeline Overview

```
Raw Data → Preprocessing → Tokenizer → Model Training → Inference / Chat
```

## Directory Structure

| Path | Description |
|------|-------------|
| `data/download_datasets.py` | Download Hindi corpora from HuggingFace (IITB parallel, Sangraha, Indic-Instruct) |
| `src/preprocess.py` | Clean & normalize JSONL datasets — Unicode NFC, noise removal, Hindi validation |
| `src/tokenizer.py` | Train 32K BPE SentencePiece tokenizer on cleaned Hindi text |
| `src/model.py` | LLaMA-style decoder-only transformer — RoPE, RMSNorm, GQA, SwiGLU |
| `src/train.py` | Training loop — AdamW, cosine LR, gradient accumulation, AMP, checkpointing |
| `inference/chat.py` | Interactive Hindi/English chat CLI + single-prompt + batch inference |
| `configs/train_config.yaml` | All training hyperparameters with scaling guide |
| `eval/` | Evaluation scripts and metrics |
| `notebooks/` | Jupyter notebooks for exploration |

## Quickstart

### 1. Download datasets
```bash
python data/download_datasets.py
```

### 2. Preprocess
```bash
python src/preprocess.py
```

### 3. Train tokenizer
```bash
python src/tokenizer.py
```

### 4. Train model
```bash
python src/train.py --batch_size 8 --max_steps 10000
# Resume from checkpoint:
python src/train.py --resume_from checkpoints/step_005000.pt
```

### 5. Chat / Inference
```bash
# Interactive chat (Hindi + English):
python inference/chat.py --checkpoint checkpoints/step_010000.pt

# Single prompt:
python inference/chat.py --checkpoint checkpoints/step_010000.pt \
    --prompt "भारत के बारे में बताओ"

# Batch inference:
python inference/chat.py --checkpoint checkpoints/step_010000.pt \
    --batch_file prompts.txt --output_file results/output.jsonl
```

## Model Architecture

- **Type**: Decoder-only causal transformer (LLaMA-style)
- **Normalization**: RMSNorm (pre-norm)
- **Positional Encoding**: RoPE (Rotary Position Embeddings)
- **Attention**: Grouped Query Attention (8Q / 4KV heads)
- **FFN**: SwiGLU activation
- **Default size**: ~25M params (scales to 120M / 350M via config)

## Scaling Guide

| Size | `dim` | `layers` | `heads` | Params |
|------|-------|----------|---------|--------|
| Small | 512 | 8 | 8 | ~25M |
| Medium | 768 | 12 | 12 | ~120M |
| Large | 1024 | 24 | 16 | ~350M |

## Requirements

```
torch >= 2.0
sentencepiece
datasets (HuggingFace)
```

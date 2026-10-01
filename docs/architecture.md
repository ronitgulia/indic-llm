# Indic LLM System Architecture

## 1. System Topology Overview
```text
  +-------------------------------------------------------------------------+
  |                           User / Client Tier                            |
  |  Web UI Playground (SPA)  |  Python Client SDK  |  REST Endpoints       |
  +-------------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------------+
  |                         Serving & Routing Tier                          |
  |  IndicRequestHandler  |  CORS & Error Handler  |  Health Monitor        |
  +-------------------------------------------------------------------------+
            |                                         |
            v                                         v
  +-------------------------+               +-------------------------------+
  |  Safety & Guardrails    |               |  Multi-Script Transliteration |
  |  - Toxicity Filter      |               |  - Devanagari <-> Bengali     |
  |  - Hallucination Guard  |               |  - Telugu <-> Tamil <-> Latin |
  +-------------------------+               +-------------------------------+
            |                                         |
            +--------------------+--------------------+
                                 v
  +-------------------------------------------------------------------------+
  |                          Generation Core Tier                           |
  |  Target Model (Llama-style Backbone)  <--  Draft Model (Speculative)    |
  |  INT8 / INT4 Quantized Kernels        <--  ONNX Runtime Edge Engine     |
  |  Indic RAG Retriever (Dense + BM25)   <--  Multi-Script Chunk Store     |
  +-------------------------------------------------------------------------+
```

## 2. Core Subsystems
- **Byte-Fallback Tokenizer:** 32,000 vocab handling unsegmented Indic scripts with zero OOV tokens.
- **LoRA PEFT:** Low-rank adapters on Q, K, V, O projections for parameter-efficient domain tuning.
- **Speculative Engine:** Verification-based draft acceleration achieving 1.8x to 2.4x speedups.
- **Quantization & ONNX:** 8-bit and 4-bit weight compression with cross-platform ONNX graph export.
- **RAG Engine:** Hybrid BM25 and dense bi-encoder indexing with script-aware document chunking.

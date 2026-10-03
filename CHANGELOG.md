# Changelog

All notable changes to the Indic LLM project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-04
### Added
- Complete Speculative Decoding engine with Draft/Target model synchronization and acceptance sampling.
- Indic Safety Guardrails & Content Moderation framework for 6 Indic languages.
- Multi-Script Unicode Transliteration across Devanagari, Bengali, Telugu, Tamil, Gujarati, and Latin.
- Quantization suite supporting INT8 and INT4 weight-only quantization and ONNX graph export.
- Synthetic Instruction Evolution engine (Evol-Instruct for Indic languages) with cultural anchor injection.
- Standalone Web UI Playground & REST API server with zero external web framework dependencies.
- Production multi-stage Dockerfile and docker-compose deployment orchestrations.
- Extended Evaluation Suite implementing BLEU-4, ChrF++, IndicReasoning (IndicMath), and Transliteration Accuracy.
- Comprehensive architectural documentation, API references, and benchmark leaderboards.

## [0.5.0] - 2026-08-04
### Added
- Indic Retrieval-Augmented Generation (RAG) system with BM25 and dense bi-encoder retrieval.
- Sentence-boundary aware text chunker for Indic punctuation marks.

## [0.4.0] - 2026-08-02
### Added
- Automated MCQA Benchmark evaluation suite covering MMLU-Indic, ARC, and Hellaswag.
- Parameter-efficient LoRA fine-tuning modules and adapter merging logic.

## [0.1.0] - 2026-07-11
### Added
- Byte-fallback BPE Tokenizer with 32,000 vocabulary supporting unsegmented Indic scripts.
- Foundational Transformer decoder backbone with RoPE and SwiGLU activations.

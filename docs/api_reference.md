# Indic LLM Public API Reference

## 1. Inference & Speculative Decoding
- `SpeculativeDecoder(draft_model, target_model, gamma=4)`: Co-generative draft-target speculative decoding engine.
- `SpeculativeGenerationConfig(gamma, temperature, top_k, top_p)`: Runtime parameter dataclass.

## 2. Safety & Guardrails
- `IndicSafetyFilter(config_path=None)`: Multilingual toxic content filter and safety screener.
- `HallucinationDetector()`: Contextual consistency verification between retrieved context and output.

## 3. Transliteration
- `IndicTransliterator()`: Fast rule-based Unicode transliterator supporting Devanagari, Bengali, Telugu, Tamil, Gujarati, and Latin scripts.

## 4. Retrieval-Augmented Generation (RAG)
- `IndicTextChunker(chunk_size, chunk_overlap)`: Script-aware sentence boundary preservation chunker.
- `BM25Retriever()`: Lexical BM25 ranker for Indic vocabularies.
- `DenseRetriever(embedding_model)`: Bi-encoder dense semantic retriever.
- `IndicRAGEngine(retriever, generator)`: End-to-end question answering pipeline.

## 5. Web Serving & SDK
- `IndicRequestHandler`: Pure standard library HTTP server handling REST endpoints.
- `IndicLLMClient(base_url)`: High-level Python client for API interaction.

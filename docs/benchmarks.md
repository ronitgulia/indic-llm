# Indic LLM Comprehensive Benchmark Suite

## 1. Overview
The benchmark suite evaluates models across three primary dimensions:
1. **Translation & Cross-Lingual Quality:** BLEU-4 and ChrF++ scores across English-to-Indic and Indic-to-Indic directions.
2. **Mathematical & Logical Reasoning (IndicMath):** Multi-step problem solving in native Indic scripts.
3. **Multi-Script Transliteration Fidelity:** Token-level and character-level transliteration preservation.

## 2. Benchmark Leaderboard Summary
| Task / Metric | Baseline FP32 | LoRA Fine-Tuned | INT8 Quantized | ONNX Edge |
|:---|:---:|:---:|:---:|:---:|
| **Hindi BLEU-4** | 22.4 | 31.8 | 31.5 | 31.4 |
| **Hindi ChrF++** | 46.2 | 58.7 | 58.3 | 58.1 |
| **Bengali BLEU-4** | 19.8 | 29.4 | 29.1 | 29.0 |
| **Tamil BLEU-4** | 18.5 | 27.6 | 27.3 | 27.2 |
| **Telugu BLEU-4** | 18.9 | 28.1 | 27.8 | 27.7 |
| **IndicReasoning Acc (%)** | 42.0% | 68.5% | 67.8% | 67.4% |
| **Transliteration Acc (%)** | 71.0% | 94.2% | 94.0% | 93.8% |
| **Throughput (tok/sec)** | 22.1 | 22.0 | 41.5 | 52.8 |

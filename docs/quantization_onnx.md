# Indic LLM — Quantization & ONNX Deployment Guide

## 1. Overview
Deploying Indic LLM models on edge devices and CPUs requires minimizing memory footprint and compute demands:
- **INT8 Weight-Only Quantization:** 2x memory reduction with negligible (<0.1%) perplexity change.
- **INT4 Packed Weight-Only Quantization:** ~3.8x memory reduction with packed 4-bit nibbles.
- **ONNX Graph Export:** Cross-platform CPU graph execution via ONNX Runtime with multi-threaded operator acceleration.

# Indic Synthetic Data Generation & Instruction Evolution

## 1. Overview
High-quality instruction tuning for Indic languages is challenged by the scarcity of publicly available, syntactically clean task datasets. This module implements an **Evol-Instruct** framework tailored for Indic NLP:
1. **Instruction Mutation:** Expanding seed prompts through cultural concretization, constraint deepening, and multi-step reasoning.
2. **Cross-Lingual Alignment:** Synthesizing bilingual terminology pairs to bridge English technical literature and Indic conceptual understanding.
3. **Rigorous Quality Filtering:** Enforcing script purity, length limits, n-gram non-repetition, and automated safety screening.

## 2. Mutation Operators
- **Concretization:** Replaces abstract queries with specific Indian cultural, geographical, agricultural, or historical anchors.
- **Deepening Constraints:** Adds multi-part output constraints (e.g. 3-point breakdown, definitions, structured summary).
- **Reasoning Expansion:** Directs the model towards step-by-step logical decomposition (Chain-of-Thought).
- **Cross-Lingual Synthesis:** Prompts the generation of native text annotated with English technical counterparts.


## 3. Dataset Generation Tutorial
To generate a batch of synthetic samples using the CLI:
```bash
python data/generate_synthetic.py --count 100 --languages hi bn ta te --output data/synthetic_100.jsonl
```

### Configuration Options
See `configs/synthetic_config.yaml` for parameters:
- `mutation_distribution`: Relative probabilities for each mutation type.
- `filtering.max_ngram_repetition_ratio`: Rejection threshold for repetitive degeneration.
- `filtering.min_quality_score`: Minimum automated quality score.

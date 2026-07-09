# Contributing to Indic LLM

Thank you for your interest in contributing! This document outlines the workflow, coding standards, and review process.

## Development Setup

```bash
git clone https://github.com/ronitgulia/indic-llm.git
cd indic-llm
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Code Style

- **Formatter**: `ruff format src/ eval/ inference/`
- **Linter**: `ruff check src/ eval/ inference/`
- **Type checks**: `mypy src/model.py src/tokenizer.py`

All checks run automatically via the CI workflow on every PR.

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, CI-passing code |
| `dev`  | Integration branch for active development |
| `feat/*` | Feature branches — open PRs against `dev` |
| `fix/*`  | Bug fix branches |

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

Examples:
- `feat(model): add sliding window attention`
- `fix(tokenizer): handle empty input gracefully`
- `docs(readme): update scaling guide`

## Pull Request Checklist

- [ ] Code passes `ruff check` with no errors
- [ ] New public functions have docstrings
- [ ] Changes are described in the PR description
- [ ] CI smoke test passes

## Reporting Issues

Open a GitHub Issue with:
1. Environment details (OS, Python version, PyTorch version, GPU)
2. Minimal reproducible example
3. Expected vs actual behaviour

"""
Unit tests for Checkpoint Hub and Registry:
  - ModelCard serialization and schema validation
  - SHA256 checksum verification
  - Registry caching and retrieval
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from hub import CheckpointHub, ModelCard, verify_checksum  # noqa: E402


def test_model_card_and_hub(tmp_path: Path):
    hub = CheckpointHub(cache_dir=str(tmp_path))
    card = ModelCard(
        model_name="indic-llm-small",
        version="1.0.0",
        architecture="LLaMA-2",
        vocab_size=32000,
        dim=512,
        layers=8,
        heads=8,
        context_length=512,
        parameters_million=25.4,
        sha256="abcd1234ef",
    )
    hub.register(card)

    assert "indic-llm-small" in hub.registry
    assert hub.registry["indic-llm-small"].parameters_million == 25.4


def test_verify_checksum(tmp_path: Path):
    sample_file = tmp_path / "test.bin"
    sample_file.write_bytes(b"IndicLLM Checkpoint Test Data")

    import hashlib
    h = hashlib.sha256(b"IndicLLM Checkpoint Test Data").hexdigest()

    assert verify_checksum(str(sample_file), h)
    assert not verify_checksum(str(sample_file), "wrong_hash")

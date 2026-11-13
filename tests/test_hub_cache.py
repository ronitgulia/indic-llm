"""
Unit tests for Checkpoint Hub cache handling, registry persistence, and checksum integrity.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from hub import CheckpointHub, ModelCard, verify_checksum  # noqa: E402


def test_hub_cache_persistence_across_instances(tmp_path: Path):
    cache_dir = str(tmp_path / "hub_cache")
    hub1 = CheckpointHub(cache_dir=cache_dir)
    card = ModelCard(
        model_name="indic-llm-125m",
        version="1.0.0",
        architecture="LLaMA-2",
        vocab_size=32000,
        dim=768,
        layers=12,
        heads=12,
        context_length=1024,
        parameters_million=125.0,
        sha256="1234567890abcdef",
        license="Apache-2.0",
    )
    hub1.register(card)
    assert "indic-llm-125m" in hub1.registry

    # Load from a fresh hub instance pointing to the same cache dir
    hub2 = CheckpointHub(cache_dir=cache_dir)
    assert "indic-llm-125m" in hub2.registry
    loaded_card = hub2.registry["indic-llm-125m"]
    assert loaded_card.dim == 768
    assert loaded_card.parameters_million == 125.0
    assert loaded_card.license == "Apache-2.0"


def test_verify_checksum_edge_cases(tmp_path: Path):
    # Non-existent file
    assert not verify_checksum(str(tmp_path / "does_not_exist.pt"), "some_hash")

    # Valid file with correct hash
    sample_path = tmp_path / "model.pt"
    data = b"Weight checkpoint bytes 123456789"
    sample_path.write_bytes(data)
    correct_hash = hashlib.sha256(data).hexdigest()

    assert verify_checksum(str(sample_path), correct_hash)
    assert verify_checksum(str(sample_path), correct_hash.upper())  # Case-insensitive

    # Corrupt / mismatch hash
    wrong_hash = hashlib.sha256(b"Tampered data").hexdigest()
    assert not verify_checksum(str(sample_path), wrong_hash)

"""
Indic LLM — Checkpoint Hub & Registry
======================================
Manages model metadata cards, checkpoint versioning, and SHA-256 verification:
  - ModelCard      : Metadata describing architecture, pretraining data, and licenses
  - verify_checksum: Cryptographic SHA256 validation for downloaded model weights
  - CheckpointHub  : Local registry and cache management for Indic LLM checkpoints
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("hub")


@dataclass
class ModelCard:
    model_name: str
    version: str
    architecture: str
    vocab_size: int
    dim: int
    layers: int
    heads: int
    context_length: int
    parameters_million: float
    sha256: str
    license: str = "MIT"
    metadata: Dict[str, Any] = field(default_factory=dict)


def verify_checksum(file_path: str, expected_sha256: str) -> bool:
    if not os.path.isfile(file_path):
        return False
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower() == expected_sha256.lower()


class CheckpointHub:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "checkpoints/hub")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.cache_dir / "registry.json"
        self.registry: Dict[str, ModelCard] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.is_file():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for name, card_dict in data.items():
                        self.registry[name] = ModelCard(**card_dict)
            except Exception as e:
                log.warning("Could not read registry: %s", e)

    def register(self, card: ModelCard) -> None:
        self.registry[card.model_name] = card
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.registry.items()}, f, indent=2)
        log.info("Registered model card: %s (v%s)", card.model_name, card.version)

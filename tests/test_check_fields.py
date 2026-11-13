"""Unit tests for dataset schema inspector utility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from check_fields import inspect_fields, main


def test_inspect_fields_success(tmp_path: Path):
    data_file = tmp_path / "test_sample.jsonl"
    records = [
        {"id": "1", "instruction": "नमस्ते", "output": "नमस्कार"},
        {"id": "2", "instruction": "भारत", "output": "देश"},
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    samples = inspect_fields(str(data_file), limit=2)
    assert len(samples) == 2
    assert "instruction" in samples[0]
    assert "output" in samples[0]


def test_inspect_fields_missing_file(tmp_path: Path):
    samples = inspect_fields(str(tmp_path / "non_existent.jsonl"))
    assert samples == []


def test_inspect_fields_cli(tmp_path: Path):
    data_file = tmp_path / "cli_sample.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"test": "value"}) + "\n")

    ret = main(["--input", str(data_file), "--limit", "1"])
    assert ret == 0


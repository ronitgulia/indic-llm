"""Unit tests for dataset download utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from download_datasets import download_hindi_data, download_indic_instruct, download_sangraha, main


@patch("datasets.load_dataset")
def test_download_hindi_data(mock_load, tmp_path: Path):
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 5
    mock_load.return_value = mock_ds

    save_dir = str(tmp_path / "raw")
    path = download_hindi_data(save_dir=save_dir)

    assert Path(path).name == "hindi_english_parallel.jsonl"
    assert Path(save_dir).exists()
    mock_ds.to_json.assert_called_once()


@patch("datasets.load_dataset")
def test_download_indic_instruct(mock_load, tmp_path: Path):
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 10
    mock_load.return_value = mock_ds

    save_dir = str(tmp_path / "custom_raw")
    path = download_indic_instruct(save_dir=save_dir)

    assert Path(path).name == "indic_instruct_hi.jsonl"
    assert Path(save_dir).exists()
    mock_ds.to_json.assert_called_once()


@patch("datasets.load_dataset")
def test_download_sangraha(mock_load, tmp_path: Path):
    mock_samples = [{"text": f"नमुना {i}"} for i in range(5)]
    mock_load.return_value = iter(mock_samples)

    save_dir = str(tmp_path / "sangraha_raw")
    path = download_sangraha(save_dir=save_dir, max_samples=3)

    assert Path(path).exists()
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


@patch("download_datasets.download_hindi_data")
def test_download_cli(mock_download_hindi, tmp_path: Path):
    main(["--dataset", "hindi", "--save_dir", str(tmp_path)])
    mock_download_hindi.assert_called_once_with(str(tmp_path))


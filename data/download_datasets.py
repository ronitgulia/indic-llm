"""
Indic LLM — Dataset Download Utilities
======================================
Automates fetching and local caching of Indic pre-training and instruction datasets:
  - IITB English-Hindi Parallel Corpus (cfilt/iitb-english-hindi)
  - Indic-Instruct Dataset (ai4bharat/indic-instruct-data-v0.1)
  - Sangraha Hindi Monolingual Verified Corpus (ai4bharat/sangraha)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")


def download_hindi_data(save_dir: str = "data/raw") -> str:
    """Download IITB English-Hindi parallel training corpus to JSONL."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hindi_english_parallel.jsonl")

    log.info("Downloading IITB Hindi-English parallel corpus...")
    from datasets import load_dataset
    dataset = load_dataset("cfilt/iitb-english-hindi", split="train")
    dataset.to_json(save_path, lines=True)
    log.info("Saved parallel dataset -> %s (%d samples)", save_path, len(dataset))
    return save_path


def download_indic_instruct(save_dir: str = "data/raw") -> str:
    """Download Indic instruction-tuning dataset to JSONL."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "indic_instruct_hi.jsonl")

    log.info("Downloading Indic instruct dataset (Anudesh)...")
    from datasets import load_dataset
    dataset = load_dataset(
        "ai4bharat/indic-instruct-data-v0.1",
        "anudesh",
        split="hi",
    )
    dataset.to_json(save_path, lines=True)
    log.info("Saved instruct dataset -> %s (%d samples)", save_path, len(dataset))
    return save_path


def download_sangraha(save_dir: str = "data/raw", max_samples: int = 50000) -> str:
    """Stream and save verified Sangraha monolingual Hindi subset to JSONL."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sangraha_hi.jsonl")

    log.info("Downloading Sangraha Hindi subset (max %d samples)...", max_samples)
    from datasets import load_dataset
    dataset = load_dataset(
        "ai4bharat/sangraha",
        "verified",
        split="hin",
        streaming=True,
    )

    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break
    log.info("Saved Sangraha dataset -> %s (%d samples)", save_path, count)
    return save_path


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Download Indic LLM pre-training and instruction datasets")
    parser.add_argument("--save_dir", type=str, default="data/raw", help="Target directory for raw JSONL files")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "hindi", "instruct", "sangraha"],
        help="Which dataset to download",
    )
    parser.add_argument("--limit", type=int, default=50000, help="Sample limit for streaming Sangraha download")
    args = parser.parse_args(argv)

    if args.dataset in ("all", "hindi"):
        download_hindi_data(args.save_dir)
    if args.dataset in ("all", "instruct"):
        download_indic_instruct(args.save_dir)
    if args.dataset in ("all", "sangraha"):
        download_sangraha(args.save_dir, max_samples=args.limit)
    log.info("Done! Datasets saved under %s", args.save_dir)


if __name__ == "__main__":
    main()

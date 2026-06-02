"""
Step 2: Dataset download script for Indic languages
Sources: AI4Bharat (updated dataset names 2024)
"""

import os
import json
from datasets import load_dataset

def download_hindi_data(save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    print("\n Downloading Hindi corpus...")
    dataset = load_dataset(
        "cfilt/iitb-english-hindi",
        split="train"
    )
    save_path = os.path.join(save_dir, "hindi_english_parallel.jsonl")
    dataset.to_json(save_path)
    print(f"Saved → {save_path} ({len(dataset)} samples)")


def download_indic_instruct(save_dir="data/raw"):
    print("\n Downloading Indic instruct dataset...")
    dataset = load_dataset(
        "ai4bharat/indic-instruct-data-v0.1",
        "anudesh",
        split="hi"
    )
    save_path = os.path.join(save_dir, "indic_instruct_hi.jsonl")
    dataset.to_json(save_path)
    print(f"Saved → {save_path} ({len(dataset)} samples)")


def download_sangraha(save_dir="data/raw"):
    print("\n Downloading Sangraha (Hindi)...")
    dataset = load_dataset(
        "ai4bharat/sangraha",
        "verified",
        split="hin",
        streaming=True
    )
    save_path = os.path.join(save_dir, "sangraha_hi.jsonl")

    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if count >= 50000:
                break
    print(f"Saved → {save_path} ({count} samples)")


if __name__ == "__main__":
    download_hindi_data()
    download_indic_instruct()
    download_sangraha()
    print("\n Done! Check data/raw/ folder")
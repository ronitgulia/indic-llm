"""
Indic LLM — Dataset Schema Inspector
====================================
CLI utility to inspect record schema, top-level keys, and preview samples
from any JSONL corpus or instruction dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional


def inspect_fields(
    file_path: str,
    limit: int = 3,
    truncate_chars: int = 300,
) -> List[dict[str, Any]]:
    """
    Inspect JSONL dataset records and print schema details.

    Parameters
    ----------
    file_path      : Path to target JSONL file
    limit          : Maximum number of samples to inspect
    truncate_chars : Length limit for previewing formatted JSON

    Returns
    -------
    List of parsed sample dictionaries.
    """
    if not os.path.exists(file_path):
        print(f"Error: dataset file not found: {file_path}", file=sys.stderr)
        return []

    samples: List[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
                print(f"Sample {i+1}: {list(sample.keys())}")
                print(json.dumps(sample, ensure_ascii=False, indent=2)[:truncate_chars])
                print("---")
            except Exception as e:
                print(f"Sample {i+1}: Failed to decode JSON ({e})")

            if len(samples) >= limit:
                break

    if not samples:
        print(f"Warning: no valid JSONL records found in {file_path}")

    return samples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect JSONL dataset structure and schema fields")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/indic_instruct_hi.jsonl",
        help="Path to JSONL dataset to inspect",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of records to inspect")
    parser.add_argument("--truncate", type=int, default=300, help="Max characters to preview")
    args = parser.parse_args(argv)

    results = inspect_fields(args.input, limit=args.limit, truncate_chars=args.truncate)
    return 0 if results or not os.path.exists(args.input) else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Step 3a: Data Preprocessing for Indic languages
- Unicode normalization
- Noise removal
- Clean text extraction
"""

import json
import unicodedata
import re
import os

def normalize_unicode(text: str) -> str:
    """Unicode normalize karo — Devanagari fix"""
    return unicodedata.normalize("NFC", text)

def remove_noise(text: str) -> str:
    """Garbage text, URLs, extra spaces remove karo"""
    # URLs remove karo
    text = re.sub(r'http\S+|www\S+', '', text)
    # HTML tags remove karo
    text = re.sub(r'<.*?>', '', text)
    # Multiple spaces/newlines clean karo
    text = re.sub(r'\s+', ' ', text)
    # Special garbage characters remove karo
    text = re.sub(r'[^\u0900-\u097F\u0020-\u007E\n]', '', text)
    return text.strip()

def is_valid_hindi(text: str, min_len=10) -> bool:
    """Check karo ki text mein Hindi characters hain"""
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    return hindi_chars >= min_len

def clean_parallel_corpus(input_path: str, output_path: str):
    """hindi_english_parallel.jsonl clean karo"""
    print(f"\n Cleaning: {input_path}")
    count_in, count_out = 0, 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            count_in += 1
            try:
                sample = json.loads(line)
                # Translation data ka structure: {"translation": {"en": ..., "hi": ...}}
                hi_text = sample.get("translation", {}).get("hi", "")
                en_text = sample.get("translation", {}).get("en", "")

                hi_text = normalize_unicode(hi_text)
                hi_text = remove_noise(hi_text)

                if is_valid_hindi(hi_text):
                    clean_sample = {
                        "hi": hi_text,
                        "en": en_text.strip()
                    }
                    fout.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")
                    count_out += 1

            except Exception:
                continue

    print(f"Input: {count_in} | Output after cleaning: {count_out}")


def clean_instruct_data(input_path: str, output_path: str):
    """indic_instruct_hi.jsonl clean karo"""
    print(f"\n Cleaning: {input_path}")
    count_in, count_out = 0, 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            count_in += 1
            try:
                sample = json.loads(line)
                # Instruction data
                instruction = normalize_unicode(sample.get("instruction", ""))
                output = normalize_unicode(sample.get("output", ""))

                if len(instruction) > 5 and len(output) > 5:
                    clean_sample = {
                        "instruction": instruction.strip(),
                        "output": output.strip()
                    }
                    fout.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")
                    count_out += 1

            except Exception:
                continue

    print(f"Input: {count_in} | Output after cleaning: {count_out}")


def clean_sangraha(input_path: str, output_path: str):
    """sangraha_hi.jsonl clean karo"""
    print(f"\n Cleaning: {input_path}")
    count_in, count_out = 0, 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            count_in += 1
            try:
                sample = json.loads(line)
                text = sample.get("text", "")
                text = normalize_unicode(text)
                text = remove_noise(text)

                if is_valid_hindi(text, min_len=20):
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count_out += 1

            except Exception:
                continue

    print(f"Input: {count_in} | Output after cleaning: {count_out}")


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    clean_parallel_corpus(
        "data/raw/hindi_english_parallel.jsonl",
        "data/processed/hindi_parallel_clean.jsonl"
    )
    clean_instruct_data(
        "data/raw/indic_instruct_hi.jsonl",
        "data/processed/indic_instruct_clean.jsonl"
    )
    clean_sangraha(
        "data/raw/sangraha_hi.jsonl",
        "data/processed/sangraha_clean.jsonl"
    )

    print("\n Preprocessing done! Check data/processed/")
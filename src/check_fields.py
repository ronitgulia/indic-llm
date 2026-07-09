import json

with open("data/raw/indic_instruct_hi.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        print(f"Sample {i+1}: {list(sample.keys())}")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:300])
        print("---")
        if i >= 2:
            break

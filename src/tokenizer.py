"""
Step 3b: SentencePiece Tokenizer Training for Indic Languages
- Trains a BPE tokenizer on cleaned Hindi corpus
- Saves vocab and model to data/tokenizer/
"""

import sentencepiece as spm
import json
import os

def prepare_text_file(input_paths: list, output_path: str, max_lines: int = 500000):
    """
    SentencePiece ke liye plain text file banao
    JSON lines se sirf text extract karo
    """
    print("\n Preparing text corpus for tokenizer training...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for input_path in input_paths:
            print(f"  Reading: {input_path}")
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        sample = json.loads(line)

                        # Har dataset ka alag structure hai
                        if "text" in sample:
                            text = sample["text"]
                        elif "hi" in sample:
                            text = sample["hi"]
                        elif "instruction" in sample:
                            text = sample["instruction"] + " " + sample["output"]
                        else:
                            continue

                        text = text.strip()
                        if len(text) > 10:
                            fout.write(text + "\n")
                            count += 1

                        if count >= max_lines:
                            break

                    except Exception:
                        continue

            if count >= max_lines:
                break

    print(f"Total lines prepared: {count}")
    return output_path


def train_tokenizer(
    text_file: str,
    model_prefix: str = "data/tokenizer/indic_spm",
    vocab_size: int = 32000,
):
    """
    SentencePiece BPE tokenizer train karo
    
    vocab_size=32000 → 32K unique tokens sikhega
    character_coverage=0.9999 → almost sab Devanagari characters cover honge
    """
    print(f"\n Training SentencePiece tokenizer...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Input: {text_file}")

    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",                  # Byte Pair Encoding
        character_coverage=0.9999,         # Devanagari sab cover karo
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        user_defined_symbols=["[INST]", "[/INST]", "<|user|>", "<|assistant|>", "<|system|>"],
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )

    print(f"\n Tokenizer saved!")
    print(f"  Model: {model_prefix}.model")
    print(f"  Vocab: {model_prefix}.vocab")


def test_tokenizer(model_path: str):
    """Tokenizer test karo — dekhte hain kaise kaam karta hai"""
    print("\n Testing tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    test_sentences = [
        "नमस्ते, आप कैसे हैं?",
        "मैं एक AI भाषा मॉडल हूं।",
        "भारत एक विविधताओं से भरा देश है।",
        "Hello, how are you?",          # English bhi test karo
        "नमस्ते Hello mixed text",      # Code-switching
    ]

    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)
        decoded = sp.decode(tokens)
        print(f"\n Input   : {sentence}")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print(f"  Decoded: {decoded}")

    print(f"\n Vocab size: {sp.get_piece_size()}")


if __name__ == "__main__":
    # Step 1: Text file prepare karo
    text_file = prepare_text_file(
        input_paths=[
            "data/processed/sangraha_clean.jsonl",
            "data/processed/hindi_parallel_clean.jsonl",
            "data/processed/indic_instruct_clean.jsonl",
        ],
        output_path="data/tokenizer/corpus.txt",
        max_lines=500000
    )

    # Step 2: Tokenizer train karo
    train_tokenizer(
        text_file=text_file,
        model_prefix="data/tokenizer/indic_spm",
        vocab_size=32000
    )

    # Step 3: Test karo
    test_tokenizer("data/tokenizer/indic_spm.model")
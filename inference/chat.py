"""
Indic LLM - Inference & Interactive Chat Engine
=================================================
Features:
  - Load model from checkpoint with automatic config detection
  - SentencePiece tokenizer integration
  - Interactive CLI chat with Hindi/English support
  - Instruction-following mode (uses [INST] / [/INST] format)
  - Batch inference for offline text generation
  - Configurable sampling: temperature, top-p, top-k, repetition penalty
  - CPU & GPU support with automatic device detection
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm
import torch

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import IndicLLM, ModelConfig

# ─────────────────────────────────────────
#  ANSI colours for terminal UX
# ─────────────────────────────────────────

class Colors:
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"


def cprint(text: str, color: str = Colors.RESET, bold: bool = False):
    prefix = (Colors.BOLD if bold else "") + color
    print(f"{prefix}{text}{Colors.RESET}")


# ─────────────────────────────────────────
#  Model Loader
# ─────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "auto") -> tuple[IndicLLM, ModelConfig]:
    """
    Load IndicLLM from a .pt checkpoint file.
    Restores model config stored inside the checkpoint.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cprint(f"Loading checkpoint: {checkpoint_path}", Colors.DIM)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct ModelConfig from saved training config
    saved_cfg = ckpt.get("model_config", {})
    model_cfg = ModelConfig(
        vocab_size  = saved_cfg.get("vocab_size", 32000),
        dim         = saved_cfg.get("model_dim", 512),
        n_layers    = saved_cfg.get("n_layers", 8),
        n_heads     = saved_cfg.get("n_heads", 8),
        n_kv_heads  = saved_cfg.get("n_kv_heads", 4),
        max_seq_len = saved_cfg.get("max_seq_len", 512),
        dropout     = 0.0,   # no dropout at inference
    )

    model = IndicLLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = model.num_parameters(trainable_only=False)
    cprint(f"Model loaded ({n_params/1e6:.1f}M params) on {device}", Colors.GREEN)
    return model, model_cfg


# ─────────────────────────────────────────
#  Repetition Penalty
# ─────────────────────────────────────────

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor,
                              penalty: float = 1.2) -> torch.Tensor:
    """
    Penalise tokens that have already appeared in the generated sequence.
    penalty > 1.0 reduces repetition; 1.0 = no effect.
    """
    if penalty == 1.0 or generated_ids.numel() == 0:
        return logits

    unique_ids = generated_ids.unique()
    for token_id in unique_ids:
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits


# ─────────────────────────────────────────
#  Core Generation Function
# ─────────────────────────────────────────

@torch.no_grad()
def generate(
    model: IndicLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    eos_id: Optional[int] = None,
    device: str = "cpu",
) -> str:
    """
    Generate text from a text prompt string.
    Returns the generated text (excluding the prompt).
    """
    import torch.nn.functional as F

    model.eval()
    if eos_id is None:
        eos_id = tokenizer.eos_id()   # typically 3

    input_ids = tokenizer.encode(prompt, out_type=int)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = tokens.clone()

    start = time.time()
    new_token_count = 0

    for _ in range(max_new_tokens):
        # Truncate context to model's max_seq_len
        ctx = tokens if tokens.shape[1] <= model.config.max_seq_len \
              else tokens[:, -model.config.max_seq_len:]

        logits, _ = model(ctx)
        logits = logits[:, -1, :].float()     # (1, vocab_size)

        # Apply repetition penalty on already-generated tokens
        logits = apply_repetition_penalty(logits, generated_ids[0], repetition_penalty)

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_to_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)

        tokens = torch.cat([tokens, next_token], dim=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        new_token_count += 1

        if next_token.item() == eos_id:
            break

    elapsed = time.time() - start
    tok_per_sec = new_token_count / max(elapsed, 1e-6)

    # Decode only the new tokens (not the prompt)
    new_ids = tokens[0, len(input_ids):].tolist()
    output_text = tokenizer.decode(new_ids)
    return output_text, tok_per_sec


# ─────────────────────────────────────────
#  Instruction Formatting
# ─────────────────────────────────────────

def format_instruct_prompt(instruction: str, system_prompt: Optional[str] = None) -> str:
    """
    Wrap user instruction in the [INST] format used during training.
    Matches the format in indic_instruct_clean.jsonl.
    """
    if system_prompt:
        return f"<|system|> {system_prompt} <|user|> {instruction} <|assistant|>"
    return f"[INST] {instruction} [/INST]"


# ─────────────────────────────────────────
#  Batch Inference
# ─────────────────────────────────────────

def batch_inference(
    model: IndicLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompts: List[str],
    output_file: str,
    **gen_kwargs,
):
    """
    Run generation on a list of prompts and save results to a JSONL file.
    Useful for offline evaluation and benchmarking.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    cprint(f"\nRunning batch inference on {len(prompts)} prompts...", Colors.CYAN)

    results = []
    for i, prompt in enumerate(prompts, 1):
        cprint(f"  [{i}/{len(prompts)}] {prompt[:60]}...", Colors.DIM)
        output, tok_s = generate(model, tokenizer, prompt, **gen_kwargs)
        results.append({
            "prompt": prompt,
            "output": output,
            "tokens_per_sec": round(tok_s, 2),
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cprint(f"\nBatch results saved → {output_file}", Colors.GREEN)
    return results


# ─────────────────────────────────────────
#  Interactive Chat CLI
# ─────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║          इंडिक LLM — इंटरैक्टिव चैट              ║
║          Indic LLM Interactive Chat Engine          ║
╠══════════════════════════════════════════════════════╣
║  Type your message in Hindi or English              ║
║  Commands:  /clear  /settings  /exit                ║
╚══════════════════════════════════════════════════════╝
"""

DEFAULT_SYSTEM = (
    "आप एक सहायक AI हैं जो हिंदी और अंग्रेज़ी दोनों में "
    "सटीक और उपयोगी जवाब देते हैं।"
)


def interactive_chat(model: IndicLLM, tokenizer: spm.SentencePieceProcessor,
                     args: argparse.Namespace, device: str):
    """Full interactive REPL for chatting with the Indic LLM."""
    cprint(BANNER, Colors.CYAN, bold=True)

    history: List[dict] = []
    system_prompt = DEFAULT_SYSTEM

    gen_kwargs = dict(
        max_new_tokens     = args.max_new_tokens,
        temperature        = args.temperature,
        top_p              = args.top_p,
        top_k              = args.top_k,
        repetition_penalty = args.repetition_penalty,
        device             = device,
    )

    while True:
        try:
            cprint("\nYou › ", Colors.YELLOW + Colors.BOLD, bold=False)
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            cprint("\n\nGoodbye! / अलविदा!", Colors.CYAN)
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────
        if user_input.lower() == "/exit":
            cprint("Goodbye! / अलविदा!", Colors.CYAN)
            break

        elif user_input.lower() == "/clear":
            history.clear()
            cprint("  Chat history cleared.", Colors.DIM)
            continue

        elif user_input.lower() == "/settings":
            cprint("\n  Current generation settings:", Colors.CYAN)
            for k, v in gen_kwargs.items():
                if k != "device":
                    print(f"    {k:25s}: {v}")
            continue

        elif user_input.startswith("/temperature "):
            try:
                gen_kwargs["temperature"] = float(user_input.split()[1])
                cprint(f"  Temperature set to {gen_kwargs['temperature']}", Colors.GREEN)
            except (IndexError, ValueError):
                cprint("  Usage: /temperature 0.7", Colors.RED)
            continue

        # ── Format prompt ──────────────────────────────────
        prompt = format_instruct_prompt(user_input, system_prompt)
        history.append({"role": "user", "content": user_input})

        # ── Generate ───────────────────────────────────────
        cprint("\nModel › ", Colors.GREEN + Colors.BOLD, bold=False)
        try:
            response, tok_per_sec = generate(model, tokenizer, prompt, **gen_kwargs)
            response = response.strip()

            # Stream-style print (character by character for UX feel)
            for char in response:
                print(char, end="", flush=True)
            print()

            history.append({"role": "assistant", "content": response})
            cprint(f"\n  [{tok_per_sec:.1f} tok/s]", Colors.DIM)

        except Exception as e:
            cprint(f"\n  Error during generation: {e}", Colors.RED)


# ─────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Indic LLM — Inference & Interactive Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat (requires checkpoint):
  python inference/chat.py --checkpoint checkpoints/step_010000.pt

  # One-shot generation:
  python inference/chat.py --checkpoint checkpoints/step_010000.pt \\
      --prompt "भारत के बारे में बताओ"

  # Batch inference from file:
  python inference/chat.py --checkpoint checkpoints/step_010000.pt \\
      --batch_file prompts.txt --output_file results/output.jsonl
        """
    )

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str,
                        default="data/tokenizer/indic_spm.model",
                        help="Path to SentencePiece .model file")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])

    # Generation params
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)

    # Modes
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for non-interactive generation")
    parser.add_argument("--batch_file", type=str, default=None,
                        help="Text file with one prompt per line (batch mode)")
    parser.add_argument("--output_file", type=str, default="results/batch_output.jsonl",
                        help="Output file for batch mode results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Device ────────────────────────────────────────────
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
             else (args.device if args.device != "auto" else "cpu")

    # ── Load tokenizer ────────────────────────────────────
    if not os.path.exists(args.tokenizer):
        cprint(f"Tokenizer not found: {args.tokenizer}", Colors.RED)
        cprint("Run: python src/tokenizer.py", Colors.YELLOW)
        sys.exit(1)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer)
    cprint(f"Tokenizer loaded — vocab size: {tokenizer.get_piece_size()}", Colors.GREEN)

    gen_kwargs = dict(
        max_new_tokens     = args.max_new_tokens,
        temperature        = args.temperature,
        top_p              = args.top_p,
        top_k              = args.top_k,
        repetition_penalty = args.repetition_penalty,
        device             = device,
    )

    # ── Load model (optional in demo mode) ────────────────
    model = None
    if args.checkpoint:
        model, _ = load_model(args.checkpoint, device)
    else:
        cprint("No checkpoint provided — running in demo/test mode (random weights)", Colors.YELLOW)
        cprint("Provide --checkpoint path to use a trained model\n", Colors.DIM)
        # Build model with random weights for sanity testing
        cfg = ModelConfig()
        model = IndicLLM(cfg).to(device)
        model.eval()

    # ── Run mode ──────────────────────────────────────────
    if args.batch_file:
        # Batch inference mode
        with open(args.batch_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        batch_inference(model, tokenizer, prompts, args.output_file, **gen_kwargs)

    elif args.prompt:
        # Single prompt mode
        cprint(f"\nPrompt: {args.prompt}", Colors.YELLOW)
        cprint("─" * 50, Colors.DIM)
        response, tok_per_sec = generate(model, tokenizer, args.prompt, **gen_kwargs)
        cprint(response.strip(), Colors.GREEN)
        cprint(f"\n[{tok_per_sec:.1f} tok/s]", Colors.DIM)

    else:
        # Interactive chat mode
        interactive_chat(model, tokenizer, args, device)

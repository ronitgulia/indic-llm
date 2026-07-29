"""
Indic LLM — Lightweight HTTP API Server
=========================================
OpenAI-compatible HTTP serving endpoint for IndicLLM:
  - POST /v1/chat/completions : Streaming (SSE) and batch chat completions
  - POST /v1/completions      : Raw prompt completions
  - GET  /v1/models           : Model metadata
  - GET  /health              : Health check endpoint

Zero external server dependencies required (uses standard library ThreadingHTTPServer).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import sentencepiece as spm
import torch

# Ensure src/ and inference/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from model import IndicLLM, ModelConfig
except ImportError:
    from src.model import IndicLLM, ModelConfig  # type: ignore[no-redef]

from chat import format_instruct_prompt, load_model
from engine import GenerationConfig, InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serve")

GLOBAL_ENGINE: Optional[InferenceEngine] = None
GLOBAL_MODEL_NAME: str = "indic-llm-base"


class LLMRequestHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible HTTP requests for IndicLLM."""

    server_version = "IndicLLMServer/1.0"

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond_json(HTTPStatus.OK, {"status": "healthy", "model": GLOBAL_MODEL_NAME})
        elif self.path in ("/v1/models", "/models"):
            self._respond_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": GLOBAL_MODEL_NAME,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "indic-llm",
                        }
                    ],
                },
            )
        else:
            self._respond_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

    def do_POST(self) -> None:
        if GLOBAL_ENGINE is None:
            self._respond_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Model engine is not initialized"},
            )
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._respond_json(
                HTTPStatus.BAD_REQUEST, {"error": "Empty request body"}
            )
            return

        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload: Dict[str, Any] = json.loads(body)
        except json.JSONDecodeError:
            self._respond_json(
                HTTPStatus.BAD_REQUEST, {"error": "Malformed JSON payload"}
            )
            return

        if self.path == "/v1/chat/completions":
            self._handle_chat_completions(payload)
        elif self.path == "/v1/completions":
            self._handle_text_completions(payload)
        else:
            self._respond_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})

    def _extract_gen_config(self, payload: Dict[str, Any]) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=payload.get("max_tokens", 256),
            temperature=payload.get("temperature", 0.8),
            top_p=payload.get("top_p", 0.92),
            top_k=payload.get("top_k", 50),
            repetition_penalty=payload.get("repetition_penalty", 1.15),
            stop_sequences=payload.get("stop", []) if isinstance(payload.get("stop"), list) else ([payload["stop"]] if payload.get("stop") else []),
        )

    def _handle_chat_completions(self, payload: Dict[str, Any]) -> None:
        messages: List[Dict[str, str]] = payload.get("messages", [])
        if not messages:
            self._respond_json(
                HTTPStatus.BAD_REQUEST, {"error": "Missing 'messages' list in request"}
            )
            return

        # Build prompt from messages
        system_prompt = None
        user_content = ""
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_content = content

        prompt = format_instruct_prompt(user_content, system_prompt)
        config = self._extract_gen_config(payload)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        stream = payload.get("stream", False)
        if stream:
            self._stream_chat_response(req_id, created, prompt, config)
        else:
            assert GLOBAL_ENGINE is not None
            text, _ = GLOBAL_ENGINE.generate(prompt, config)
            resp = {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": GLOBAL_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text.strip()},
                        "finish_reason": "stop",
                    }
                ],
            }
            self._respond_json(HTTPStatus.OK, resp)

    def _stream_chat_response(
        self, req_id: str, created: int, prompt: str, config: GenerationConfig
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._send_cors_headers()
        self.end_headers()

        assert GLOBAL_ENGINE is not None
        for chunk_text, _ in GLOBAL_ENGINE.generate_stream(prompt, config):
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": GLOBAL_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            msg = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()

        # Send final chunk with finish_reason
        final_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": GLOBAL_MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_text_completions(self, payload: Dict[str, Any]) -> None:
        prompt = payload.get("prompt", "")
        if not prompt:
            self._respond_json(
                HTTPStatus.BAD_REQUEST, {"error": "Missing 'prompt' string in request"}
            )
            return

        config = self._extract_gen_config(payload)
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        assert GLOBAL_ENGINE is not None
        text, _ = GLOBAL_ENGINE.generate(prompt, config)
        resp = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": GLOBAL_MODEL_NAME,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
        self._respond_json(HTTPStatus.OK, resp)

    def _respond_json(self, status: HTTPStatus, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), format % args)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    checkpoint: Optional[str] = None,
    tokenizer_path: str = "data/tokenizer/indic_spm.model",
    device: str = "auto",
) -> None:
    """Start the HTTP server on specified host and port."""
    global GLOBAL_ENGINE, GLOBAL_MODEL_NAME

    resolved_device = (
        "cuda" if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    if checkpoint and os.path.exists(checkpoint):
        model, _ = load_model(checkpoint, device=resolved_device)
        GLOBAL_MODEL_NAME = Path(checkpoint).stem
    else:
        log.warning("No checkpoint provided — starting in demo mode with random weights")
        cfg = ModelConfig()
        model = IndicLLM(cfg).to(resolved_device)
        GLOBAL_MODEL_NAME = "indic-llm-demo"

    GLOBAL_ENGINE = InferenceEngine(model, tokenizer, device=torch.device(resolved_device))

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, LLMRequestHandler)
    log.info("Indic LLM API server listening on http://%s:%d", host, port)
    log.info("Endpoints: /v1/chat/completions | /v1/completions | /v1/models | /health")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Server shutting down...")
        httpd.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indic LLM API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/indic_spm.model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

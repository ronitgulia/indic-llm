"""Lightweight Python client SDK for Indic LLM Web API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class IndicLLMClient:
    """Client for querying the Indic LLM HTTP service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        """Check server health status."""
        url = f"{self.base_url}/api/health"
        req = urllib.request.Request(url, headers={"User-Agent": "IndicLLMClient/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(
        self,
        prompt: str,
        language: str = "hi",
        temperature: float = 0.7,
        enable_guardrails: bool = True,
        use_speculative: bool = False,
    ) -> dict[str, Any]:
        """Request text completion."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "prompt": prompt,
            "language": language,
            "temperature": temperature,
            "enable_guardrails": enable_guardrails,
            "use_speculative": use_speculative,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "IndicLLMClient/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

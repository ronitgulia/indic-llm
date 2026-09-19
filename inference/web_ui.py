"""Lightweight HTTP Web UI & REST API for Indic LLM Playground.

Provides a responsive single-page browser interface and REST endpoints
for real-time Indic text generation, chat, transliteration, and safety inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("indic_llm.web")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indic LLM Playground</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --text-muted: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 2rem; max-width: 1100px; margin: 0 auto; }
        header { margin-bottom: 2rem; text-align: center; }
        header h1 { font-size: 2.2rem; color: #0f172a; margin-bottom: 0.5rem; }
        header p { color: var(--text-muted); font-size: 1.1rem; }
        .grid { display: grid; grid-template-columns: 1fr 340px; gap: 1.5rem; }
        @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .control-group { margin-bottom: 1.2rem; }
        label { display: block; font-weight: 600; font-size: 0.88rem; margin-bottom: 0.4rem; color: #334155; }
        input[type="text"], textarea, select { width: 100%; border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; font-size: 0.95rem; }
        textarea { resize: vertical; min-height: 120px; }
        .btn { background: var(--primary); color: white; border: none; border-radius: 8px; padding: 0.8rem 1.5rem; font-weight: 600; cursor: pointer; width: 100%; transition: background 0.2s; }
        .btn:hover { background: var(--primary-hover); }
        .output-box { margin-top: 1.5rem; background: #f1f5f9; border-radius: 8px; padding: 1rem; min-height: 140px; white-space: pre-wrap; font-size: 0.95rem; border: 1px solid #cbd5e1; }
        .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; background: #e0f2fe; color: #0369a1; }
    </style>
</head>
<body>
    <header>
        <h1>🇮🇳 Indic LLM Playground</h1>
        <p>Real-time multilingual instruction tuning & generative playground</p>
    </header>
    <div class="grid">
        <div class="card">
            <div class="control-group">
                <label for="prompt">Input Prompt</label>
                <textarea id="prompt" placeholder="उदा. कृत्रिम बुद्धिमत्ता के लाभ समझाइए..."></textarea>
            </div>
            <button class="btn" onclick="submitPrompt()">Generate Response</button>
            <div class="output-box" id="output">Generated response will appear here...</div>
        </div>
        <div class="card">
            <h3>Generation Controls</h3>
            <div class="control-group" style="margin-top: 1rem;">
                <label for="lang">Language</label>
                <select id="lang">
                    <option value="hi">Hindi (हिन्दी)</option>
                    <option value="bn">Bengali (বাংলা)</option>
                    <option value="ta">Tamil (தமிழ்)</option>
                    <option value="te">Telugu (తెలుగు)</option>
                    <option value="gu">Gujarati (ગુજરાતી)</option>
                    <option value="mr">Marathi (मराठी)</option>
                </select>
            </div>
            <div class="control-group">
                <label for="temp">Temperature (<span id="temp-val">0.7</span>)</label>
                <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.7" oninput="document.getElementById('temp-val').innerText=this.value">
            </div>
            <div class="control-group">
                <label><input type="checkbox" id="guardrails" checked> Enable Safety Guardrails</label>
            </div>
            <div class="control-group">
                <label><input type="checkbox" id="speculative" checked> Speculative Decoding</label>
            </div>
        </div>
    </div>
    <script>
        async function submitPrompt() {
            const prompt = document.getElementById('prompt').value;
            const lang = document.getElementById('lang').value;
            const temp = parseFloat(document.getElementById('temp').value);
            const guardrails = document.getElementById('guardrails').checked;
            const speculative = document.getElementById('speculative').checked;
            const output = document.getElementById('output');
            output.innerText = "Generating response...";
            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, language: lang, temperature: temp, enable_guardrails: guardrails, use_speculative: speculative})
                });
                const data = await res.json();
                output.innerText = data.completion || JSON.stringify(data, null, 2);
            } catch (e) {
                output.innerText = "Error: " + e.message;
            }
        }
    </script>
</body>
</html>
"""

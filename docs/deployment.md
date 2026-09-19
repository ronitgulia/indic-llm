# Indic LLM Deployment & Serving Guide

## 1. Local Web Playground
The project includes a self-contained web server requiring no heavy frontend frameworks:
```bash
python inference/web_ui.py --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` in your web browser.

## 2. REST Endpoints
- `GET /api/health` — System status and version info.
- `POST /api/generate` — Text generation with safety and speculative decoding controls.
- `POST /api/chat` — Multi-turn conversation completion.
- `POST /api/transliterate` — Script conversion across Indic scripts.

"""Unit tests for Indic Web UI & REST API server."""

import json
from io import BytesIO
from unittest.mock import MagicMock
from inference.web_ui import IndicRequestHandler


def _make_handler(method: str, path: str, body: dict | None = None):
    handler = IndicRequestHandler.__new__(IndicRequestHandler)
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.headers = {}
    handler.wfile = BytesIO()

    if body is not None:
        raw_body = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(raw_body))
        handler.headers["Content-Type"] = "application/json"
        handler.rfile = BytesIO(raw_body)
    else:
        handler.rfile = BytesIO()

    # Mock send_response, send_header, end_headers
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    return handler


def test_health_endpoint():
    handler = _make_handler("GET", "/api/health")
    handler.do_GET()
    handler.send_response.assert_called_with(200)
    output = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert output["status"] == "healthy"
    assert output["service"] == "indic-llm-web"


def test_generate_endpoint_success():
    payload = {"prompt": "नमस्ते", "language": "hi", "temperature": 0.5}
    handler = _make_handler("POST", "/api/generate", payload)
    handler.do_POST()
    handler.send_response.assert_called_with(200)
    output = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "नमस्ते" in output["completion"]
    assert output["language"] == "hi"


def test_generate_missing_prompt():
    payload = {"language": "hi"}
    handler = _make_handler("POST", "/api/generate", payload)
    handler.do_POST()
    handler.send_response.assert_called_with(400)
    output = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "error" in output


def test_not_found_endpoint():
    handler = _make_handler("GET", "/api/unknown")
    handler.do_GET()
    handler.send_response.assert_called_with(404)


def test_chat_endpoint():
    payload = {"messages": [{"role": "user", "content": "नमस्ते"}]}
    handler = _make_handler("POST", "/api/chat", payload)
    handler.do_POST()
    handler.send_response.assert_called_with(200)
    output = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "reply" in output
    assert output["messages_count"] == 2


def test_transliterate_endpoint():
    payload = {"text": "namaste", "target_script": "Devanagari"}
    handler = _make_handler("POST", "/api/transliterate", payload)
    handler.do_POST()
    handler.send_response.assert_called_with(200)
    output = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert output["target_script"] == "Devanagari"

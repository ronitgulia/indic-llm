"""Unit tests for IndicLLMClient SDK."""

from unittest.mock import MagicMock, patch
from inference.client import IndicLLMClient


@patch("urllib.request.urlopen")
def test_client_health(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"status": "healthy"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    client = IndicLLMClient("http://localhost:8000")
    res = client.health()
    assert res["status"] == "healthy"


@patch("urllib.request.urlopen")
def test_client_generate(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"completion": "उत्तर", "language": "hi"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    client = IndicLLMClient("http://localhost:8000")
    res = client.generate("प्रश्न")
    assert res["completion"] == "उत्तर"
    assert res["language"] == "hi"

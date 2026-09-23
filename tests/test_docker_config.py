"""Validation tests for Docker and container configuration files."""

from pathlib import Path


def test_dockerfile_exists_and_valid():
    df = Path("docker/Dockerfile")
    assert df.exists(), "Dockerfile must exist"
    content = df.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in content
    assert "HEALTHCHECK" in content
    assert "USER indicuser" in content


def test_docker_compose_exists_and_valid():
    dc = Path("docker-compose.yml")
    assert dc.exists(), "docker-compose.yml must exist"
    content = dc.read_text(encoding="utf-8")
    assert "services:" in content
    assert "indic-llm-web:" in content
    assert "8000:8000" in content

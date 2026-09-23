# Multi-stage production Dockerfile for Indic LLM Web Serving
FROM python:3.11-slim as builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Runner stage
FROM python:3.11-slim as runner

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN groupadd -r indic && useradd -r -g indic -d /app indicuser

COPY . /app
RUN chown -R indicuser:indic /app

USER indicuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "inference/web_ui.py", "--host", "0.0.0.0", "--port", "8000"]

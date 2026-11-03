FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/models_cache

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download model weights to prevent cold-start timeouts
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ground_truth.json data/evaluation_results.json ./data/

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

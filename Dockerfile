FROM python:3.11-slim

# Install system dependencies and certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary for reproducible and fast environment management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Environment variables for Python runtime and offline model cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/models_cache

# Copy package manifests first for optimal Docker layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies without development tooling
RUN uv sync --frozen --no-dev

# Pre-download SentenceTransformer model into Docker image to guarantee offline execution
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application source, scripts, and evaluation metadata
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ground_truth.json data/evaluation_results.json ./data/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default entrypoint starts the FastAPI production server
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

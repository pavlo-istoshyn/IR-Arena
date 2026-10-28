"""Service for text and job embeddings using SentenceTransformers."""

import logging
from functools import lru_cache
from typing import Any

from src.core.config import get_settings
from src.schemas.job import JobListing

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIMENSION = 384


class JobEmbedder:
    """Wrapper around SentenceTransformer for generating vector embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Lazy load SentenceTransformer model on first usage."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model: '%s'...", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_job_text(self, job: JobListing) -> str:
        """
        Combine job title, extracted skills, and description into a single
        semantically rich string for embedding generation.
        """
        skills_str = ", ".join(job.skills) if job.skills else ""
        parts = [job.title]
        if skills_str:
            parts.append(f"Skills: {skills_str}")
        if job.description:
            parts.append(job.description)
        return ". ".join(parts).strip()

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode a batch of texts into normalized float vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in embeddings]

    def embed_jobs(self, jobs: list[JobListing], batch_size: int = 64) -> list[list[float]]:
        """Extract combined text from jobs and compute their embeddings."""
        texts = [self.build_job_text(job) for job in jobs]
        return self.embed_texts(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> list[float]:
        """Encode a single search query into a normalized float vector."""
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else [0.0] * self.dimension


@lru_cache
def get_embedder() -> JobEmbedder:
    """Factory creating singleton JobEmbedder configured from application settings."""
    settings = get_settings()
    return JobEmbedder(
        model_name=settings.ml.default_embedding_model,
        dimension=settings.ml.embedding_dimension,
    )

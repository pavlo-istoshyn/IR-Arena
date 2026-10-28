"""Elasticsearch index management, schema mapping, and bulk indexing."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from src.core.config import get_settings
from src.repositories.base import JOBS_INDEX_NAME
from src.schemas.job import JobListing

logger = logging.getLogger(__name__)


def get_jobs_index_mapping(dims: int | None = None) -> dict[str, Any]:
    """Build Elasticsearch index mapping with dynamic vector dimension on demand."""
    vector_dims = dims if dims is not None else get_settings().ml.embedding_dimension
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "german_clean": {
                        "type": "german",
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "german_clean",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "description": {
                    "type": "text",
                    "analyzer": "german_clean",
                },
                "company": {
                    "type": "keyword",
                    "fields": {"text": {"type": "text"}},
                },
                "city": {"type": "keyword"},
                "is_remote": {"type": "boolean"},
                "skills": {
                    "type": "keyword",
                    "fields": {"text": {"type": "text"}},
                },
                "source_url": {
                    "type": "keyword",
                    "index": False,
                },
                "published_date": {
                    "type": "keyword",
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": vector_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }


class JobsIndexManager:
    """Manager responsible for Elasticsearch index lifecycle, schema, and bulk indexing."""

    def __init__(
        self,
        client: AsyncElasticsearch,
        index_name: str = JOBS_INDEX_NAME,
        dims: int | None = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.dims = dims

    @property
    def embedding_dimension(self) -> int:
        """Resolve embedding dimension lazily on demand."""
        if self.dims is not None:
            return self.dims
        return get_settings().ml.embedding_dimension

    @property
    def mapping(self) -> dict[str, Any]:
        """Index mapping with vector dimension evaluated lazily on demand."""
        return get_jobs_index_mapping(dims=self.embedding_dimension)

    async def create_index(self, recreate: bool = False) -> bool:
        """
        Create Elasticsearch index with the defined schema mapping.
        If recreate is True, drops the existing index first.
        Returns True if a new index was created, False if it already existed.
        """
        exists = bool(await self.client.indices.exists(index=self.index_name))
        if exists:
            if recreate:
                logger.info("Dropping existing index '%s'...", self.index_name)
                await self.client.indices.delete(index=self.index_name)
            else:
                logger.info("Index '%s' already exists. Keeping current schema.", self.index_name)
                return False

        mapping = self.mapping
        logger.info("Creating index '%s' with dense_vector and text mappings...", self.index_name)
        await self.client.indices.create(
            index=self.index_name,
            settings=mapping.get("settings"),
            mappings=mapping.get("mappings"),
        )
        return True

    async def generate_bulk_actions(
        self,
        jobs: list[JobListing],
        embeddings: list[list[float]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate bulk indexing actions pairing each JobListing with its embedding vector."""
        for job, emb in zip(jobs, embeddings, strict=True):
            yield {
                "_index": self.index_name,
                "_id": job.id,
                "_source": {
                    **job.model_dump(exclude={"id"}),
                    "embedding": emb,
                },
            }

    async def index_jobs_bulk(
        self,
        jobs: list[JobListing],
        embeddings: list[list[float]],
        chunk_size: int = 200,
    ) -> tuple[int, int]:
        """
        Bulk index job listings and embeddings into Elasticsearch.
        Returns a tuple of (success_count, error_count).
        """
        if len(jobs) != len(embeddings):
            raise ValueError(
                f"Jobs count ({len(jobs)}) and embeddings count ({len(embeddings)}) must match."
            )

        actions = self.generate_bulk_actions(jobs, embeddings)
        success_count, errors = await async_bulk(
            self.client,
            actions,
            chunk_size=chunk_size,
            raise_on_error=False,
        )
        error_count = len(errors) if isinstance(errors, list) else 0
        return success_count, error_count

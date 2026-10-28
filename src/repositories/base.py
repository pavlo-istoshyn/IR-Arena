"""Base repository protocols and abstractions."""

from typing import Protocol

from src.schemas.search import JobHit, SearchFilters

JOBS_INDEX_NAME = "jobs"


class JobSearchRepository(Protocol):
    """Abstraction for searching job listings using lexical and vector techniques."""

    async def search_bm25(
        self,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[JobHit]:
        """Execute BM25 lexical search."""
        ...

    async def search_knn(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[JobHit]:
        """Execute kNN vector search."""
        ...

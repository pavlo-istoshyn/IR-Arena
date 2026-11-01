"""Elasticsearch job search repository implementation."""

from typing import Any

from elasticsearch import AsyncElasticsearch

from src.repositories.base import JOBS_INDEX_NAME
from src.schemas.search import JobHit, SearchFilters


def build_search_filters(filters: SearchFilters | None = None) -> list[dict[str, Any]]:
    """Build Elasticsearch boolean filter clauses from SearchFilters."""
    if filters is None:
        return []

    clauses: list[dict[str, Any]] = []
    if filters.city:
        clauses.append({"term": {"city": filters.city}})
    if filters.is_remote is not None:
        clauses.append({"term": {"is_remote": filters.is_remote}})
    if filters.skills:
        clauses.append({"terms": {"skills": filters.skills}})
    if filters.seniority:
        clauses.append({"term": {"seniority": filters.seniority.lower()}})
    return clauses


def parse_es_hits(raw_hits: list[dict[str, Any]]) -> list[JobHit]:
    """Convert raw Elasticsearch search hits into domain JobHit objects."""
    hits: list[JobHit] = []
    for rank, hit in enumerate(raw_hits, start=1):
        source = hit.get("_source", {})
        hits.append(
            JobHit(
                id=str(hit.get("_id", "")),
                score=float(hit.get("_score") or 0.0),
                rank=rank,
                **source,
            )
        )
    return hits


class ElasticsearchJobRepository:
    """Elasticsearch repository for querying job documents and applying search filters."""

    def __init__(
        self,
        client: AsyncElasticsearch,
        index_name: str = JOBS_INDEX_NAME,
    ) -> None:
        self.client = client
        self.index_name = index_name

    async def _execute_search(
        self,
        limit: int,
        query: dict[str, Any] | None = None,
        knn: dict[str, Any] | None = None,
    ) -> list[JobHit]:
        kwargs: dict[str, Any] = {
            "index": self.index_name,
            "size": limit,
            "source_excludes": ["embedding"],
        }
        if query:
            kwargs["query"] = query
        if knn:
            kwargs["knn"] = knn

        response = await self.client.search(**kwargs)
        raw_hits = response.get("hits", {}).get("hits", [])
        return parse_es_hits(raw_hits)

    async def search_bm25(
        self,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[JobHit]:
        """Execute full-text BM25 search over title, skills, and description."""
        filter_clauses = build_search_filters(filters=filters)
        must_clause: dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "skills.text^2", "description"],
                "fuzziness": "AUTO",
            }
        }
        es_query: dict[str, Any] = (
            {"bool": {"must": must_clause, "filter": filter_clauses}}
            if filter_clauses
            else {"bool": {"must": must_clause}}
        )
        return await self._execute_search(limit=limit, query=es_query)

    async def search_knn(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[JobHit]:
        """Execute dense vector kNN search using HNSW cosine similarity."""
        filter_clauses = build_search_filters(filters=filters)
        knn_clause: dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": limit,
            "num_candidates": max(limit * 5, 50),
        }
        if filter_clauses:
            knn_clause["filter"] = (
                {"bool": {"filter": filter_clauses}}
                if len(filter_clauses) > 1
                else filter_clauses[0]
            )
        return await self._execute_search(limit=limit, knn=knn_clause)

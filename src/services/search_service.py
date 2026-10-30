"""Search service implementing BM25, kNN dense vector, and Hybrid search strategies."""

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

from src.repositories.base import JobSearchRepository
from src.schemas.search import JobHit, SearchRequest, SearchResponse
from src.services.embedder import JobEmbedder
from src.services.fusion import (
    RankFusionStrategy,
    ReciprocalRankFusion,
)

DEFAULT_HYBRID_CANDIDATE_LIMIT = 50


class SearchService:
    """Search service coordinating lexical BM25, dense vector, and hybrid rank fusion retrieval."""

    def __init__(
        self,
        repository: JobSearchRepository,
        embedder: JobEmbedder,
        fusion_strategy: RankFusionStrategy | None = None,
    ) -> None:
        self.repository: JobSearchRepository = repository
        self.embedder: JobEmbedder = embedder
        self.fusion_strategy: RankFusionStrategy = fusion_strategy or ReciprocalRankFusion()

    @staticmethod
    async def _timed_search(
        query: str,
        strategy: str,
        coro: Coroutine[Any, Any, list[JobHit]],
    ) -> SearchResponse:
        """Helper to measure search latency and package into a SearchResponse."""
        start_time = time.perf_counter()
        hits = await coro
        took_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return SearchResponse(
            query=query,
            strategy=strategy,
            total=len(hits),
            took_ms=took_ms,
            items=hits,
        )

    async def search_bm25(self, request: SearchRequest) -> SearchResponse:
        """Execute full-text BM25 search."""
        return await self._timed_search(
            query=request.query,
            strategy="bm25",
            coro=self.repository.search_bm25(
                query=request.query,
                limit=request.limit,
                filters=request.filters,
            ),
        )

    async def search_knn(self, request: SearchRequest) -> SearchResponse:
        """Execute semantic vector search with asynchronous query encoding."""

        async def _knn() -> list[JobHit]:
            query_vector = await asyncio.to_thread(self.embedder.embed_query, request.query)
            return await self.repository.search_knn(
                query_vector=query_vector,
                limit=request.limit,
                filters=request.filters,
            )

        return await self._timed_search(
            query=request.query,
            strategy="knn",
            coro=_knn(),
        )

    async def search_hybrid(
        self,
        request: SearchRequest,
        candidate_limit: int = DEFAULT_HYBRID_CANDIDATE_LIMIT,
        fusion_strategy: RankFusionStrategy | None = None,
    ) -> SearchResponse:
        """
        Execute hybrid search by running BM25 and kNN concurrently and fusing
        results using the configured or explicitly provided RankFusionStrategy.
        """
        active_strategy = fusion_strategy or self.fusion_strategy

        async def _hybrid() -> list[JobHit]:
            query_vector = await asyncio.to_thread(self.embedder.embed_query, request.query)
            fetch_k = max(request.limit * 2, candidate_limit)

            bm25_task = self.repository.search_bm25(
                query=request.query,
                limit=fetch_k,
                filters=request.filters,
            )
            knn_task = self.repository.search_knn(
                query_vector=query_vector,
                limit=fetch_k,
                filters=request.filters,
            )
            bm25_hits, knn_hits = await asyncio.gather(bm25_task, knn_task)
            return active_strategy.fuse(
                sources={"bm25": bm25_hits, "knn": knn_hits},
                limit=request.limit,
            )

        return await self._timed_search(
            query=request.query,
            strategy=f"hybrid_{active_strategy.name}",
            coro=_hybrid(),
        )

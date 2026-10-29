"""Abstract Base Class for rank fusion strategies."""

from abc import ABC, abstractmethod
from typing import Any

from src.schemas.search import JobHit


class RankFusionStrategy(ABC):
    """
    Abstract Strategy Interface for rank fusion algorithms.

    All concrete fusion strategies must implement:
    1. `name`: identifier of the strategy (e.g. 'rrf', 'weighted').
    2. `fuse`: accepts named source lists and returns a unified ranked list.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the strategy."""
        pass

    @abstractmethod
    def fuse(
        self,
        sources: dict[str, list[JobHit]],
        limit: int = 10,
    ) -> list[JobHit]:
        """
        Fuse arbitrary named ranked lists into a single ranked list.

        Args:
            sources: Mapping from retrieval source name (e.g. 'bm25', 'knn', 'reranker')
                to its ranked list of JobHit items.
            limit: Maximum number of merged results to return.

        Returns:
            list[JobHit]: Re-ranked, fused list of unique documents.
        """
        pass

    def build_ranked_hits(
        self,
        entries: list[dict[str, Any]],
        limit: int = 10,
    ) -> list[JobHit]:
        """
        Sort fused candidate entries descending by fused_score (breaking ties
        by sum of available ranks), slice to `limit`, and construct JobHit objects.
        """
        sorted_entries = sorted(
            entries,
            key=lambda e: (-e["fused_score"], sum(e["ranks"].values())),
        )

        results: list[JobHit] = []
        for rank, entry in enumerate(sorted_entries[:limit], start=1):
            orig: JobHit = entry["hit"]
            results.append(
                orig.model_copy(
                    update={
                        "score": round(entry["fused_score"], 6),
                        "rank": rank,
                        "details": {
                            "strategy": self.name,
                            "ranks": entry["ranks"],
                            "scores": entry["scores"],
                        },
                    }
                )
            )
        return results

"""Reciprocal Rank Fusion (RRF) Strategy."""

from typing import Any

from src.schemas.search import JobHit
from src.services.fusion.base import RankFusionStrategy

DEFAULT_RRF_K = 60


class ReciprocalRankFusion(RankFusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) Strategy (Cormack et al., 2009).

    RRF_Score(d) = sum_{m in models} 1 / (k + rank_m(d))

    Does not rely on score calibration; supports any number N of named sources.
    Documents appearing near the top of multiple ranking lists receive a higher combined score.
    """

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        if k < 1:
            raise ValueError(f"RRF parameter k must be >= 1, got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return "rrf"

    def fuse(
        self,
        sources: dict[str, list[JobHit]],
        limit: int = 10,
    ) -> list[JobHit]:
        """Fuse N named ranked lists using Reciprocal Rank Fusion."""
        if not sources:
            return []

        combined: dict[str, dict[str, Any]] = {}

        for source_name, hits in sources.items():
            for hit in hits:
                rank = hit.rank
                score_component = 1.0 / (self.k + rank)
                if hit.id in combined:
                    combined[hit.id]["fused_score"] += score_component
                    combined[hit.id]["ranks"][source_name] = rank
                    combined[hit.id]["scores"][source_name] = hit.score
                else:
                    combined[hit.id] = {
                        "hit": hit,
                        "fused_score": score_component,
                        "ranks": {source_name: rank},
                        "scores": {source_name: hit.score},
                    }

        return self.build_ranked_hits(list(combined.values()), limit=limit)

"""Weighted Score Fusion (WSF) Strategy with Min-Max Score Normalization."""

from typing import Any

from src.schemas.search import JobHit
from src.services.fusion.base import RankFusionStrategy


class WeightedScoreFusion(RankFusionStrategy):
    """
    Weighted Score Fusion (WSF) Strategy with Min-Max Score Normalization.

    Score(d) = sum_{m in models} weight_m * norm_score_m(d)

    Since score ranges differ across retrieval systems (e.g. BM25 unbounded vs cosine in [0, 1]),
    scores from each system are first normalized to [0, 1] using Min-Max scaling
    within the candidate window before applying linear combination.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        raw_weights = weights if weights is not None else {"bm25": 0.5, "knn": 0.5}
        if any(w < 0.0 for w in raw_weights.values()):
            raise ValueError("Weights must be non-negative")
        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            raise ValueError("Total weight must be greater than zero")
        self.weights: dict[str, float] = {k: v / total_weight for k, v in raw_weights.items()}

    @property
    def name(self) -> str:
        return "weighted"

    @staticmethod
    def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores of a set of candidates to [0, 1]."""
        if not scores:
            return {}
        min_val = min(scores.values())
        max_val = max(scores.values())
        val_range = max_val - min_val
        if val_range <= 0:
            return dict.fromkeys(scores, 1.0)
        return {k: (v - min_val) / val_range for k, v in scores.items()}

    def fuse(
        self,
        sources: dict[str, list[JobHit]],
        limit: int = 10,
    ) -> list[JobHit]:
        """Fuse N named ranked lists using normalized weighted score summation."""
        if not sources:
            return []

        source_names = list(sources.keys())
        default_weight = 1.0 / len(source_names)
        raw_w = {name: self.weights.get(name, default_weight) for name in source_names}
        total_w = sum(raw_w.values()) or 1.0
        weights_map = {name: w / total_w for name, w in raw_w.items()}

        norm_scores: dict[str, dict[str, float]] = {
            name: self._min_max_normalize({hit.id: hit.score for hit in hits})
            for name, hits in sources.items()
        }

        combined: dict[str, dict[str, Any]] = {}
        for name, hits in sources.items():
            for hit in hits:
                norm_component = weights_map[name] * norm_scores[name].get(hit.id, 0.0)
                if hit.id in combined:
                    combined[hit.id]["fused_score"] += norm_component
                    combined[hit.id]["ranks"][name] = hit.rank
                    combined[hit.id]["scores"][name] = hit.score
                else:
                    combined[hit.id] = {
                        "hit": hit,
                        "fused_score": norm_component,
                        "ranks": {name: hit.rank},
                        "scores": {name: hit.score},
                    }

        return self.build_ranked_hits(list(combined.values()), limit=limit)

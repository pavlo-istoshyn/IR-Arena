"""Rank fusion package implementing the Strategy Pattern for Search & IR."""

from src.services.fusion.base import RankFusionStrategy
from src.services.fusion.rrf import DEFAULT_RRF_K, ReciprocalRankFusion
from src.services.fusion.weighted import WeightedScoreFusion

__all__ = [
    "RankFusionStrategy",
    "ReciprocalRankFusion",
    "WeightedScoreFusion",
    "DEFAULT_RRF_K",
]

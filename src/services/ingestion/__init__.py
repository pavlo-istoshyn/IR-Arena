"""Data ingestion, filtering, normalization, and skill extraction OOP services."""

from src.services.ingestion.job_collector import DEFAULT_PROFILE_QUERIES, JobCollector
from src.services.ingestion.job_filter import JobRoleFilter
from src.services.ingestion.job_normalizer import JobNormalizer
from src.services.ingestion.seniority import detect_seniority, get_seniority_badge
from src.services.ingestion.skill_extractor import (
    REMOTE_PATTERNS,
    SKILL_PATTERNS,
    SkillExtractor,
)

__all__ = [
    "DEFAULT_PROFILE_QUERIES",
    "JobCollector",
    "JobRoleFilter",
    "JobNormalizer",
    "SkillExtractor",
    "SKILL_PATTERNS",
    "REMOTE_PATTERNS",
    "detect_seniority",
    "get_seniority_badge",
]

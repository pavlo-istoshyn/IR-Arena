"""Pydantic schemas for search requests, hits, filters, and responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.job import JobListing


class SearchFilters(BaseModel):
    """Common filtering parameters for job searches."""

    model_config = ConfigDict(str_strip_whitespace=True)

    city: str | None = None
    is_remote: bool | None = None
    skills: list[str] | None = None


class SearchRequest(SearchFilters):
    """Search request parameters across all search strategies."""

    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)

    @property
    def filters(self) -> SearchFilters:
        """Extract filters as a dedicated SearchFilters instance."""
        return SearchFilters(city=self.city, is_remote=self.is_remote, skills=self.skills)


class JobHit(JobListing):
    """Search result item extending JobListing with relevance rank, score, and fusion details."""

    score: float
    rank: int = Field(..., ge=1)
    details: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """Search response envelope returning metadata and ranked items."""

    query: str
    strategy: str
    total: int = Field(..., ge=0)
    took_ms: float = Field(..., ge=0.0)
    items: list[JobHit] = Field(default_factory=list)

"""Pydantic v2 schemas for job listings."""

from pydantic import BaseModel, ConfigDict, Field


class JobListing(BaseModel):
    """Normalized job listing schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    company: str = ""
    city: str | None = None
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    is_remote: bool = False
    source_url: str | None = None
    published_date: str | None = None
    seniority: str = "mid"

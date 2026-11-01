"""Unit tests for Pydantic v2 schemas (JobListing, SearchFilters, SearchRequest, JobHit)."""

import pytest
from pydantic import ValidationError

from src.schemas import JobHit, JobListing, SearchFilters, SearchRequest


def test_job_listing_valid_full():
    job = JobListing(
        id="10001-1003871444-S",
        title="Python Backend Developer (m/w/d)",
        company="Tech Corp GmbH",
        city="Berlin",
        description="Developing scalable backend services using FastAPI and PostgreSQL.",
        skills=["Python", "FastAPI", "PostgreSQL"],
        is_remote=True,
        source_url="https://www.arbeitsagentur.de/jobsuche/jobdetail/10001-1003871444-S",
        published_date="2026-10-20",
    )
    assert job.id == "10001-1003871444-S"
    assert job.title == "Python Backend Developer (m/w/d)"
    assert job.company == "Tech Corp GmbH"
    assert job.city == "Berlin"
    assert "FastAPI" in job.description
    assert job.skills == ["Python", "FastAPI", "PostgreSQL"]
    assert job.is_remote is True
    assert job.source_url == "https://www.arbeitsagentur.de/jobsuche/jobdetail/10001-1003871444-S"
    assert job.published_date == "2026-10-20"


def test_job_listing_defaults():
    job = JobListing(id="12345", title="Software Engineer")
    assert job.id == "12345"
    assert job.title == "Software Engineer"
    assert job.company == ""
    assert job.city is None
    assert job.description == ""
    assert job.skills == []
    assert job.is_remote is False
    assert job.source_url is None
    assert job.published_date is None
    assert job.seniority == "mid"


def test_job_listing_missing_required_fields():
    with pytest.raises(ValidationError):
        JobListing()  # type: ignore

    with pytest.raises(ValidationError):
        JobListing(id="12345")  # type: ignore

    with pytest.raises(ValidationError):
        JobListing(title="Python Dev")  # type: ignore


def test_job_listing_whitespace_stripping():
    job = JobListing(
        id="  999  ",
        title="  Backend Dev  ",
        company="  Tech GmbH  ",
        skills=["  Python  ", "FastAPI  "],
        city=" Munich ",
    )
    assert job.id == "999"
    assert job.title == "Backend Dev"
    assert job.company == "Tech GmbH"
    assert job.city == "Munich"
    assert job.skills == ["Python", "FastAPI"]


def test_job_listing_extra_fields_ignored():
    job = JobListing(
        id="123",
        title="Dev",
        unsupported_field="some_value",  # type: ignore[call-arg]
        internal_metadata={"foo": "bar"},  # type: ignore[call-arg]
    )
    assert job.id == "123"
    assert not hasattr(job, "unsupported_field")


def test_job_listing_json_roundtrip():
    original = JobListing(
        id="ref-789",
        title="Senior Python / AI Engineer",
        company="AI Labs Berlin",
        city="Berlin",
        description="Working on LLM agents and RAG pipelines.",
        skills=["Python", "LLM", "RAG", "Docker"],
        is_remote=True,
        source_url="https://example.com/jobs/789",
        published_date="2026-10-26",
    )
    json_str = original.model_dump_json()
    restored = JobListing.model_validate_json(json_str)
    assert restored == original


def test_job_listing_rejects_empty_id_or_title():
    with pytest.raises(ValidationError):
        JobListing(id="", title="Valid Title")

    with pytest.raises(ValidationError):
        JobListing(id="   ", title="Valid Title")

    with pytest.raises(ValidationError):
        JobListing(id="valid-id", title="")

    with pytest.raises(ValidationError):
        JobListing(id="valid-id", title="   ")


def test_job_hit_inherits_job_listing():
    """Verify JobHit inherits all fields and validation from JobListing without code duplication."""
    hit = JobHit(
        id="hit-1",
        title="Python Dev",
        company="Startup GmbH",
        score=0.032,
        rank=1,
    )
    assert isinstance(hit, JobListing)
    assert hit.id == "hit-1"
    assert hit.title == "Python Dev"
    assert hit.company == "Startup GmbH"
    assert hit.score == 0.032
    assert hit.rank == 1
    assert hit.details is None


def test_search_filters_and_request_extraction():
    """Verify SearchRequest inherits from SearchFilters and provides .filters property."""
    req = SearchRequest(
        query="Python FastAPI",
        limit=20,
        city="Berlin",
        is_remote=True,
        skills=["Python", "Docker"],
        seniority="junior",
    )
    assert isinstance(req, SearchFilters)
    filters = req.filters
    assert isinstance(filters, SearchFilters)
    assert filters.city == "Berlin"
    assert filters.is_remote is True
    assert filters.skills == ["Python", "Docker"]
    assert filters.seniority == "junior"

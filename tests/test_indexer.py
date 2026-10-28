"""Unit tests for JobsIndexManager and Elasticsearch index management."""

from unittest.mock import AsyncMock, patch

import pytest

from src.repositories.index_manager import JOBS_INDEX_NAME, JobsIndexManager
from src.schemas.job import JobListing


def test_jobs_index_manager_mapping_structure():
    mock_client = AsyncMock()
    manager = JobsIndexManager(client=mock_client, dims=384)

    assert manager.embedding_dimension == 384
    props = manager.mapping["mappings"]["properties"]

    assert props["title"]["type"] == "text"
    assert props["description"]["type"] == "text"
    assert props["city"]["type"] == "keyword"
    assert props["is_remote"]["type"] == "boolean"
    assert props["skills"]["type"] == "keyword"

    embedding_prop = props["embedding"]
    assert embedding_prop["type"] == "dense_vector"
    assert embedding_prop["dims"] == 384
    assert embedding_prop["index"] is True
    assert embedding_prop["similarity"] == "cosine"


def test_jobs_index_manager_custom_dimension():
    """Verify JobsIndexManager supports custom vector dimensions without globals."""
    mock_client = AsyncMock()
    manager_768 = JobsIndexManager(client=mock_client, dims=768)
    assert manager_768.embedding_dimension == 768
    props = manager_768.mapping["mappings"]["properties"]
    assert props["embedding"]["dims"] == 768


@pytest.mark.asyncio
async def test_create_jobs_index_when_not_exists():
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = False

    manager = JobsIndexManager(client=mock_client, dims=384)
    created = await manager.create_index(recreate=False)
    assert created is True
    mock_client.indices.create.assert_called_once_with(
        index=JOBS_INDEX_NAME,
        settings=manager.mapping["settings"],
        mappings=manager.mapping["mappings"],
    )


@pytest.mark.asyncio
async def test_create_jobs_index_skip_when_exists():
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = True

    manager = JobsIndexManager(client=mock_client)
    created = await manager.create_index(recreate=False)
    assert created is False
    mock_client.indices.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_jobs_index_recreate():
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = True

    manager = JobsIndexManager(client=mock_client)
    created = await manager.create_index(recreate=True)
    assert created is True
    mock_client.indices.delete.assert_called_once_with(index=JOBS_INDEX_NAME)
    mock_client.indices.create.assert_called_once()


@pytest.mark.asyncio
async def test_index_jobs_bulk_success():
    mock_client = AsyncMock()
    jobs = [
        JobListing(
            id="job-1",
            title="Python Developer",
            company="Company 1",
            city="Berlin",
            description="Backend dev",
            skills=["Python"],
            is_remote=True,
        )
    ]
    embeddings = [[0.1] * 384]

    manager = JobsIndexManager(client=mock_client)
    with patch("src.repositories.index_manager.async_bulk", new=AsyncMock(return_value=(1, []))):
        success, errors = await manager.index_jobs_bulk(jobs, embeddings)
        assert success == 1
        assert errors == 0

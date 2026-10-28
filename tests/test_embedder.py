from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.schemas.job import JobListing
from src.services.embedder import JobEmbedder


def test_build_job_text():
    embedder = JobEmbedder()
    job = JobListing(
        id="123",
        title="Python Backend Developer",
        company="Startup GmbH",
        city="Berlin",
        description="Building FastAPI microservices.",
        skills=["Python", "FastAPI", "Docker"],
        is_remote=True,
    )
    text = embedder.build_job_text(job)
    assert "Python Backend Developer" in text
    assert "Skills: Python, FastAPI, Docker" in text
    assert "Building FastAPI microservices." in text


def test_embed_texts_mocked():
    embedder = JobEmbedder()
    fake_vectors = np.array([[0.1] * 384, [-0.2] * 384])

    mock_model = MagicMock()
    mock_model.encode.return_value = fake_vectors
    embedder._model = mock_model

    vectors = embedder.embed_texts(["Text 1", "Text 2"], batch_size=2)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert isinstance(vectors[0], list)
    assert isinstance(vectors[0][0], float)
    mock_model.encode.assert_called_once_with(
        ["Text 1", "Text 2"],
        batch_size=2,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def test_embed_query_mocked():
    embedder = JobEmbedder()
    fake_vector = np.array([[0.5] * 384])

    mock_model = MagicMock()
    mock_model.encode.return_value = fake_vector
    embedder._model = mock_model

    vec = embedder.embed_query("FastAPI Python Developer")
    assert len(vec) == 384
    assert isinstance(vec, list)
    assert vec[0] == pytest.approx(0.5)

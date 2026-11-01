"""Unit and integration tests for search endpoints, service, and rank fusion strategies."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.repositories.jobs_repository import ElasticsearchJobRepository, build_search_filters
from src.routers.deps import get_search_service
from src.schemas.search import JobHit, SearchFilters, SearchRequest, SearchResponse
from src.services.fusion import (
    RankFusionStrategy,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from src.services.search_service import SearchService


def _make_dummy_hit(job_id: str, title: str, rank: int, score: float = 1.0) -> JobHit:
    """Helper to create minimal JobHit for fusion testing."""
    return JobHit(
        id=job_id,
        title=title,
        company="Tech Corp",
        city="Berlin",
        description="Developing modern software solutions with Python.",
        skills=["Python", "FastAPI"],
        is_remote=True,
        source_url="https://example.com/jobs/" + job_id,
        published_date="2026-10-01",
        score=score,
        rank=rank,
    )


def test_rrf_boosts_documents_in_both_lists():
    """A document appearing in both lists must rank higher than one at rank 1 in only one list."""
    bm25_hits = [
        _make_dummy_hit("job-A", "Python Engineer", rank=1, score=15.0),
        _make_dummy_hit("job-C", "Junior Python", rank=2, score=12.0),
    ]
    knn_hits = [
        _make_dummy_hit("job-B", "Data Scientist", rank=1, score=0.88),
        _make_dummy_hit("job-A", "Python Engineer", rank=2, score=0.85),
    ]

    rrf = ReciprocalRankFusion(k=60)
    fused = rrf.fuse({"bm25": bm25_hits, "knn": knn_hits}, limit=10)

    assert len(fused) == 3
    assert fused[0].id == "job-A"
    assert fused[0].rank == 1
    expected_score = round(1.0 / 61 + 1.0 / 62, 6)
    assert fused[0].score == expected_score
    assert fused[0].details is not None
    assert fused[0].details["ranks"] == {"bm25": 1, "knn": 2}
    assert fused[0].details["scores"] == {"bm25": 15.0, "knn": 0.85}

    assert fused[1].id == "job-B"
    assert fused[1].rank == 2
    assert fused[1].score == round(1.0 / 61, 6)

    assert fused[2].id == "job-C"
    assert fused[2].rank == 3
    assert fused[2].score == round(1.0 / 62, 6)


def test_rrf_empty_inputs():
    """Empty inputs should return an empty list gracefully."""
    rrf = ReciprocalRankFusion()
    assert rrf.fuse({}) == []
    assert rrf.fuse({"bm25": [], "knn": []}) == []


def test_rrf_one_empty_list():
    """When only one strategy produces hits, RRF scales correctly."""
    hits = [_make_dummy_hit("job-1", "Dev", rank=1)]
    fused = ReciprocalRankFusion(k=60).fuse({"bm25": hits, "knn": []})
    assert len(fused) == 1
    assert fused[0].id == "job-1"
    assert fused[0].score == round(1.0 / 61, 6)
    assert fused[0].details is not None
    assert fused[0].details["ranks"] == {"bm25": 1}


def test_rrf_limit_respected():
    """RRF must truncate results to the requested limit."""
    bm25 = [_make_dummy_hit(f"j-{i}", f"Title {i}", rank=i) for i in range(1, 20)]
    fused = ReciprocalRankFusion().fuse({"bm25": bm25, "knn": []}, limit=5)
    assert len(fused) == 5
    assert [h.rank for h in fused] == [1, 2, 3, 4, 5]


def test_rrf_class_instance_and_validation():
    """Test ReciprocalRankFusion class initialization and validation."""
    rrf = ReciprocalRankFusion(k=30)
    assert rrf.k == 30

    bm25 = [_make_dummy_hit("job-1", "Python", rank=1)]
    knn = [_make_dummy_hit("job-1", "Python", rank=1)]
    fused = rrf.fuse({"bm25": bm25, "knn": knn})
    assert len(fused) == 1
    assert fused[0].score == round(2.0 / 31, 6)

    with pytest.raises(ValueError, match="RRF parameter k must be >= 1"):
        ReciprocalRankFusion(k=0)


def test_rank_fusion_strategy_cannot_be_instantiated():
    """Verify that RankFusionStrategy is an abstract interface."""
    with pytest.raises(TypeError):
        RankFusionStrategy()  # type: ignore[abstract]


def test_weighted_score_fusion_strategy():
    """Test WeightedScoreFusion Min-Max normalization and weight combination."""
    wsf = WeightedScoreFusion(weights={"bm25": 0.7, "knn": 0.3})
    assert round(wsf.weights["bm25"], 2) == 0.7
    assert round(wsf.weights["knn"], 2) == 0.3

    bm25_hits = [
        _make_dummy_hit("job-1", "FastAPI Engineer", rank=1, score=20.0),
        _make_dummy_hit("job-2", "Python Dev", rank=2, score=10.0),
    ]
    knn_hits = [
        _make_dummy_hit("job-2", "Python Dev", rank=1, score=0.9),
        _make_dummy_hit("job-1", "FastAPI Engineer", rank=2, score=0.8),
    ]

    fused = wsf.fuse({"bm25": bm25_hits, "knn": knn_hits}, limit=10)
    assert len(fused) == 2
    assert fused[0].details is not None
    assert fused[0].details["strategy"] == "weighted"

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        WeightedScoreFusion(weights={"bm25": -0.1, "knn": 0.5})

    with pytest.raises(ValueError, match="Total weight must be greater than zero"):
        WeightedScoreFusion(weights={"bm25": 0.0, "knn": 0.0})


def test_build_search_filters():
    assert build_search_filters() == []
    assert build_search_filters(SearchFilters(city="München")) == [{"term": {"city": "München"}}]
    assert build_search_filters(SearchFilters(is_remote=True)) == [{"term": {"is_remote": True}}]
    assert build_search_filters(SearchFilters(skills=["Python", "Docker"])) == [
        {"terms": {"skills": ["Python", "Docker"]}}
    ]
    assert build_search_filters(SearchFilters(seniority="junior")) == [
        {"term": {"seniority": "junior"}}
    ]

    all_filters = build_search_filters(
        SearchFilters(city="Berlin", is_remote=False, skills=["FastAPI"], seniority="senior")
    )
    assert len(all_filters) == 4
    assert {"term": {"seniority": "senior"}} in all_filters


@pytest.mark.asyncio
async def test_search_bm25_repository():
    mock_es = AsyncMock()
    mock_es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "10001",
                    "_score": 4.5,
                    "_source": {
                        "title": "Backend Python Developer",
                        "company": "Startup XYZ",
                        "city": "Berlin",
                        "description": "Building microservices.",
                        "skills": ["Python", "FastAPI"],
                        "is_remote": True,
                        "source_url": "https://example.com/10001",
                        "published_date": "2026-10-10",
                    },
                }
            ]
        }
    }

    repo = ElasticsearchJobRepository(client=mock_es)
    hits = await repo.search_bm25(
        query="python backend", limit=5, filters=SearchFilters(city="Berlin")
    )
    assert len(hits) == 1
    assert hits[0].id == "10001"
    assert hits[0].title == "Backend Python Developer"
    assert hits[0].score == 4.5
    assert hits[0].rank == 1

    mock_es.search.assert_awaited_once()
    call_kwargs = mock_es.search.call_args.kwargs
    assert call_kwargs["index"] == "jobs"
    assert call_kwargs["size"] == 5
    assert call_kwargs["source_excludes"] == ["embedding"]
    assert "bool" in call_kwargs["query"]


@pytest.mark.asyncio
async def test_search_knn_repository():
    mock_es = AsyncMock()
    mock_es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "10002",
                    "_score": 0.89,
                    "_source": {
                        "title": "AI Search Engineer",
                        "company": "Deep Tech",
                        "city": "Hamburg",
                        "description": "Vector search systems.",
                        "skills": ["Python", "Elasticsearch"],
                        "is_remote": False,
                    },
                }
            ]
        }
    }

    dummy_vector = [0.1] * 384
    repo = ElasticsearchJobRepository(client=mock_es)
    hits = await repo.search_knn(
        query_vector=dummy_vector, limit=5, filters=SearchFilters(is_remote=False)
    )
    assert len(hits) == 1
    assert hits[0].id == "10002"
    assert hits[0].score == 0.89
    assert hits[0].rank == 1

    mock_es.search.assert_awaited_once()
    call_kwargs = mock_es.search.call_args.kwargs
    assert call_kwargs["index"] == "jobs"
    assert call_kwargs["size"] == 5
    assert call_kwargs["source_excludes"] == ["embedding"]
    assert "knn" in call_kwargs
    assert call_kwargs["knn"]["field"] == "embedding"


@pytest.mark.asyncio
async def test_search_service_methods():
    mock_repo = AsyncMock()
    mock_repo.search_bm25.return_value = [_make_dummy_hit("j1", "Python Dev", rank=1)]
    mock_repo.search_knn.return_value = [_make_dummy_hit("j2", "ML Dev", rank=1)]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.05] * 384

    service = SearchService(repository=mock_repo, embedder=mock_embedder)

    res_bm25 = await service.search_bm25(SearchRequest(query="python", limit=5))
    assert res_bm25.strategy == "bm25"
    assert res_bm25.total == 1
    assert res_bm25.items[0].id == "j1"
    assert res_bm25.took_ms >= 0.0

    res_knn = await service.search_knn(SearchRequest(query="machine learning", limit=5))
    assert res_knn.strategy == "knn"
    assert res_knn.total == 1
    assert res_knn.items[0].id == "j2"
    mock_embedder.embed_query.assert_called_with("machine learning")

    res_hybrid = await service.search_hybrid(SearchRequest(query="python ml", limit=5))
    assert res_hybrid.strategy == "hybrid_rrf"
    assert res_hybrid.total == 2
    assert {res_hybrid.items[0].id, res_hybrid.items[1].id} == {"j1", "j2"}


@pytest.mark.asyncio
async def test_api_search_bm25_endpoint():
    mock_service = AsyncMock()
    dummy_response = SearchResponse(
        query="python",
        strategy="bm25",
        total=1,
        took_ms=12.5,
        items=[_make_dummy_hit("job-101", "Senior Python Engineer", rank=1, score=9.2)],
    )
    mock_service.search_bm25.return_value = dummy_response

    app.dependency_overrides[get_search_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/search/bm25",
                json={"query": "python", "limit": 10, "city": "Berlin", "is_remote": True},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "bm25"
        assert data["total"] == 1
        assert data["items"][0]["id"] == "job-101"
        assert data["items"][0]["score"] == 9.2
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_search_knn_endpoint():
    mock_service = AsyncMock()
    dummy_response = SearchResponse(
        query="vector search",
        strategy="knn",
        total=1,
        took_ms=25.0,
        items=[_make_dummy_hit("job-102", "Vector Search Engineer", rank=1, score=0.91)],
    )
    mock_service.search_knn.return_value = dummy_response

    app.dependency_overrides[get_search_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/search/knn",
                json={"query": "vector search", "limit": 5},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "knn"
        assert data["items"][0]["id"] == "job-102"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_search_hybrid_endpoint():
    mock_service = AsyncMock()
    dummy_response = SearchResponse(
        query="fastapi backend",
        strategy="hybrid_rrf",
        total=1,
        took_ms=30.0,
        items=[_make_dummy_hit("job-103", "FastAPI Engineer", rank=1, score=0.0325)],
    )
    mock_service.search_hybrid.return_value = dummy_response

    app.dependency_overrides[get_search_service] = lambda: mock_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/search/hybrid",
                json={"query": "fastapi backend", "limit": 10},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "hybrid_rrf"
        assert data["items"][0]["id"] == "job-103"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_search_validation_errors():
    """Ensure invalid payloads return 422 Unprocessable Entity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_empty = await client.post("/search/bm25", json={"query": ""})
        assert res_empty.status_code == 422

        res_missing = await client.post("/search/knn", json={"limit": 10})
        assert res_missing.status_code == 422

        res_limit_zero = await client.post("/search/hybrid", json={"query": "test", "limit": 0})
        assert res_limit_zero.status_code == 422

        res_limit_huge = await client.post("/search/hybrid", json={"query": "test", "limit": 200})
        assert res_limit_huge.status_code == 422


def test_n_list_fusion_support():
    """Verify that fusion strategies support N named sources and preserve generic details."""
    list1 = [_make_dummy_hit("j1", "Python Dev", rank=1, score=10.0)]
    list2 = [
        _make_dummy_hit("j1", "Python Dev", rank=2, score=0.8),
        _make_dummy_hit("j2", "ML Eng", rank=1, score=0.9),
    ]
    list3 = [
        _make_dummy_hit("j1", "Python Dev", rank=1, score=0.95),
        _make_dummy_hit("j3", "Data Eng", rank=2, score=0.7),
    ]

    sources = {"bm25": list1, "knn": list2, "reranker": list3}

    rrf = ReciprocalRankFusion(k=60)
    rrf_hits = rrf.fuse(sources, limit=5)
    assert len(rrf_hits) == 3
    assert rrf_hits[0].id == "j1"
    assert rrf_hits[0].details is not None
    assert rrf_hits[0].details["ranks"] == {"bm25": 1, "knn": 2, "reranker": 1}
    assert rrf_hits[0].details["scores"]["reranker"] == 0.95

    weighted = WeightedScoreFusion(weights={"bm25": 0.3, "knn": 0.3, "reranker": 0.4})
    weighted_hits = weighted.fuse(sources, limit=5)
    assert len(weighted_hits) == 3
    assert weighted_hits[0].id == "j1"
    assert weighted_hits[0].details is not None
    assert weighted_hits[0].details["ranks"] == {"bm25": 1, "knn": 2, "reranker": 1}


@pytest.mark.asyncio
async def test_search_service_open_closed_principle():
    """Verify that adding a new fusion strategy requires no changes to SearchService (Open/Closed)."""

    class CustomRerankStrategy(RankFusionStrategy):
        @property
        def name(self) -> str:
            return "cross_encoder"

        def fuse(self, sources: dict[str, list[JobHit]], limit: int = 10) -> list[JobHit]:
            first_list = next(iter(sources.values())) if sources else []
            entries = [
                {
                    "hit": h,
                    "fused_score": 0.99,
                    "ranks": {"cross_encoder": 1},
                    "scores": {"cross_encoder": 0.99},
                }
                for h in first_list
            ]
            return self.build_ranked_hits(entries, limit=limit)

    mock_repo = AsyncMock()
    mock_repo.search_bm25.return_value = [_make_dummy_hit("j1", "Python", rank=1)]
    mock_repo.search_knn.return_value = [_make_dummy_hit("j2", "FastAPI", rank=1)]
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    service = SearchService(
        repository=mock_repo,
        embedder=mock_embedder,
        fusion_strategy=CustomRerankStrategy(),
    )

    response = await service.search_hybrid(SearchRequest(query="python"))
    assert response.strategy == "hybrid_cross_encoder"
    assert response.total == 1
    assert response.items[0].details == {
        "strategy": "cross_encoder",
        "ranks": {"cross_encoder": 1},
        "scores": {"cross_encoder": 0.99},
    }

"""Unit tests for IRMetricsCalculator, SearchEvaluator, and ground-truth schema."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.schemas.evaluation import BenchmarkQuery
from src.schemas.search import JobHit, SearchFilters, SearchResponse
from src.services.evaluation import IRMetricsCalculator, SearchEvaluator
from src.services.fusion import ReciprocalRankFusion


def _make_hit(job_id: str, rank: int, score: float = 1.0) -> JobHit:
    return JobHit(
        id=job_id,
        title=f"Job {job_id}",
        company="Test GmbH",
        score=score,
        rank=rank,
    )


def test_benchmark_query_inherits_search_filters():
    bq = BenchmarkQuery(
        id="Q01",
        query="Python FastAPI",
        city="Berlin",
        is_remote=True,
        skills=["Python"],
        relevant_docs={"j1": 3},
    )
    assert isinstance(bq, SearchFilters)
    assert bq.query == "Python FastAPI"
    assert bq.city == "Berlin"
    assert bq.is_remote is True
    assert bq.skills == ["Python"]


def test_metrics_calculator_edge_cases():
    calc = IRMetricsCalculator()
    assert calc.precision_at_k([], {"d1": 1}, k=5) == 0.0
    assert calc.precision_at_k(["d1"], {"d1": 1}, k=0) == 0.0
    assert calc.recall_at_k(["d1"], {}, k=5) == 0.0
    assert calc.recall_at_k(["d1"], {"d1": 1}, k=0) == 0.0
    assert calc.ndcg_at_k(["d1"], {"d1": 1}, k=0) == 0.0
    assert calc.dcg_at_k([1], k=0) == 0.0


def test_precision_at_k():
    calc = IRMetricsCalculator()
    rel = {"d1": 3, "d2": 2, "d3": 1}
    retrieved = ["d1", "x1", "d2", "x2", "x3"]
    assert calc.precision_at_k(retrieved, rel, k=5) == 0.4
    assert calc.precision_at_k(retrieved, rel, k=2) == 0.5
    assert calc.precision_at_k([], rel, k=5) == 0.0
    assert calc.precision_at_k(retrieved, rel, k=0) == 0.0


def test_recall_at_k():
    calc = IRMetricsCalculator()
    rel = {"d1": 3, "d2": 2, "d3": 1, "d4": 1}
    retrieved = ["d1", "x1", "d2", "x2"]
    assert calc.recall_at_k(retrieved, rel, k=4) == 0.5
    assert calc.recall_at_k(retrieved, {}, k=4) == 0.0
    assert calc.recall_at_k([], rel, k=4) == 0.0


def test_reciprocal_rank():
    calc = IRMetricsCalculator()
    rel = {"d2": 3, "d4": 1}
    assert calc.reciprocal_rank(["d2", "d1", "d4"], rel) == 1.0
    assert calc.reciprocal_rank(["x1", "x2", "d4", "d2"], rel) == round(1.0 / 3, 4)
    assert calc.reciprocal_rank(["x1", "x2"], rel) == 0.0
    assert calc.reciprocal_rank(["x1", "x2", "d2"], rel, k=2) == 0.0


def test_dcg_and_ndcg_at_k():
    calc = IRMetricsCalculator()
    rel = {"d1": 3, "d2": 2, "d3": 1}

    ideal_retrieved = ["d1", "d2", "d3", "x1"]
    assert calc.ndcg_at_k(ideal_retrieved, rel, k=4) == 1.0

    suboptimal = ["x1", "d3", "d2", "d1"]
    score_sub = calc.ndcg_at_k(suboptimal, rel, k=4)
    assert 0.0 < score_sub < 1.0

    assert calc.ndcg_at_k(["x1", "x2"], rel, k=4) == 0.0
    assert calc.dcg_at_k([], k=4) == 0.0
    assert calc.ndcg_at_k([], rel, k=4) == 0.0


def test_percentile_calculation():
    calc = IRMetricsCalculator()
    assert calc.percentile([], 50.0) == 0.0
    assert calc.percentile([12.345], 95.0) == 12.35
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calc.percentile(vals, 50.0) == 30.0
    assert calc.percentile(vals, 0.0) == 10.0
    assert calc.percentile(vals, 100.0) == 50.0


@pytest.mark.asyncio
async def test_search_evaluator_all_strategies_and_ocp():
    mock_service = AsyncMock()
    mock_service.search_bm25.return_value = SearchResponse(
        query="python",
        strategy="bm25",
        total=2,
        took_ms=8.0,
        items=[_make_hit("j2", rank=1), _make_hit("j1", rank=2)],
    )
    mock_service.search_knn.return_value = SearchResponse(
        query="python",
        strategy="knn",
        total=2,
        took_ms=15.0,
        items=[_make_hit("j1", rank=1), _make_hit("x1", rank=2)],
    )
    mock_service.search_hybrid.return_value = SearchResponse(
        query="python",
        strategy="hybrid_rrf",
        total=2,
        took_ms=20.0,
        items=[_make_hit("j1", rank=1), _make_hit("j2", rank=2)],
    )

    evaluator = SearchEvaluator(search_service=mock_service)
    queries = [
        BenchmarkQuery(
            id="Q01",
            query="python",
            relevant_docs={"j1": 3, "j2": 2},
        )
    ]

    reports = await evaluator.evaluate_all(queries=queries, k=2)
    assert len(reports) == 4
    by_strategy = {r.strategy: r for r in reports}

    assert by_strategy["hybrid_rrf"].mean_ndcg_at_k == 1.0
    assert by_strategy["hybrid_rrf"].mean_mrr == 1.0
    assert by_strategy["hybrid_rrf"].mean_recall_at_k == 1.0

    assert by_strategy["bm25"].mean_ndcg_at_k < 1.0
    assert by_strategy["bm25"].mean_recall_at_k == 1.0

    # Test Open/Closed Principle: custom fusion strategy registry without modifying SearchEvaluator
    custom_evaluator = SearchEvaluator(
        search_service=mock_service,
        fusion_strategies={"custom_rrf_k20": ReciprocalRankFusion(k=20)},
    )
    custom_reports = await custom_evaluator.evaluate_all(queries=queries, k=2)
    assert [r.strategy for r in custom_reports] == ["bm25", "knn", "custom_rrf_k20"]

    with pytest.raises(ValueError, match="Unsupported evaluation strategy"):
        await evaluator.evaluate_strategy(queries=queries, strategy="unknown_strategy", k=2)


def test_ground_truth_dataset_valid():
    """Ensure data/ground_truth.json exists and validates cleanly against BenchmarkQuery."""
    gt_path = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
    assert gt_path.exists()
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    queries = [BenchmarkQuery.model_validate(item) for item in raw]
    assert len(queries) == 50
    for q in queries:
        assert len(q.relevant_docs) > 0
        assert all(rel in (1, 2, 3) for rel in q.relevant_docs.values())


def test_dashboard_helpers():
    """Verify dashboard data loaders and DataFrame builder against evaluation results."""
    from scripts.dashboard import (
        build_metrics_dataframe,
        load_benchmark_queries,
        load_evaluation_reports,
    )

    queries = load_benchmark_queries()
    assert len(queries) == 50

    reports = load_evaluation_reports()
    assert len(reports) == 4

    df = build_metrics_dataframe(reports)
    assert len(df) == 4
    assert "Strategy" in df.columns
    assert "NDCG@10" in df.columns
    assert "MRR" in df.columns

"""Pydantic v2 schemas for IR evaluation benchmarks and reports."""

from pydantic import BaseModel, Field

from src.schemas.search import SearchFilters


class BenchmarkQuery(SearchFilters):
    """Single benchmark query: search text, optional filters, and graded relevance judgments."""

    id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    description: str = ""
    relevant_docs: dict[str, int] = Field(default_factory=dict)


class QueryEvaluationResult(BaseModel):
    """IR metrics and latency computed for a single query under a specific strategy."""

    query_id: str
    query: str
    precision_at_k: float = Field(..., ge=0.0, le=1.0)
    recall_at_k: float = Field(..., ge=0.0, le=1.0)
    mrr: float = Field(..., ge=0.0, le=1.0)
    ndcg_at_k: float = Field(..., ge=0.0, le=1.0)
    took_ms: float = Field(..., ge=0.0)
    retrieved_ids: list[str] = Field(default_factory=list)


class StrategyEvaluationReport(BaseModel):
    """Aggregated evaluation report for a single search strategy across all benchmark queries."""

    strategy: str
    k: int = Field(..., ge=1)
    num_queries: int = Field(..., ge=0)
    mean_precision_at_k: float = Field(..., ge=0.0, le=1.0)
    mean_recall_at_k: float = Field(..., ge=0.0, le=1.0)
    mean_mrr: float = Field(..., ge=0.0, le=1.0)
    mean_ndcg_at_k: float = Field(..., ge=0.0, le=1.0)
    mean_latency_ms: float = Field(..., ge=0.0)
    p50_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    query_results: list[QueryEvaluationResult] = Field(default_factory=list)

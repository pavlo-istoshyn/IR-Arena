"""Search evaluator orchestrating benchmark runs across retrieval and fusion strategies."""

from src.schemas.evaluation import (
    BenchmarkQuery,
    QueryEvaluationResult,
    StrategyEvaluationReport,
)
from src.schemas.search import SearchRequest, SearchResponse
from src.services.evaluation.metrics import IRMetricsCalculator
from src.services.fusion import (
    RankFusionStrategy,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from src.services.search_service import SearchService


class SearchEvaluator:
    """Evaluates search strategies against a ground-truth benchmark dataset."""

    def __init__(
        self,
        search_service: SearchService,
        metrics_calculator: IRMetricsCalculator | None = None,
        fusion_strategies: dict[str, RankFusionStrategy] | None = None,
    ) -> None:
        self.search_service = search_service
        self.metrics = metrics_calculator or IRMetricsCalculator()
        self.fusion_strategies: dict[str, RankFusionStrategy] = (
            fusion_strategies
            if fusion_strategies is not None
            else {
                "hybrid_rrf": ReciprocalRankFusion(),
                "hybrid_weighted": WeightedScoreFusion(),
            }
        )

    def evaluate_single_query(
        self,
        benchmark_query: BenchmarkQuery,
        response: SearchResponse,
        k: int = 10,
    ) -> QueryEvaluationResult:
        """Compute IR metrics for a single query response against ground-truth relevance."""
        retrieved_ids = [hit.id for hit in response.items]
        rel_docs = benchmark_query.relevant_docs

        return QueryEvaluationResult(
            query_id=benchmark_query.id,
            query=benchmark_query.query,
            precision_at_k=self.metrics.precision_at_k(retrieved_ids, rel_docs, k=k),
            recall_at_k=self.metrics.recall_at_k(retrieved_ids, rel_docs, k=k),
            mrr=self.metrics.reciprocal_rank(retrieved_ids, rel_docs, k=k),
            ndcg_at_k=self.metrics.ndcg_at_k(retrieved_ids, rel_docs, k=k),
            took_ms=response.took_ms,
            retrieved_ids=retrieved_ids[:k],
        )

    async def _execute_strategy(
        self,
        request: SearchRequest,
        strategy: str,
    ) -> SearchResponse:
        if strategy == "bm25":
            return await self.search_service.search_bm25(request)
        if strategy == "knn":
            return await self.search_service.search_knn(request)
        if strategy in self.fusion_strategies:
            return await self.search_service.search_hybrid(
                request,
                fusion_strategy=self.fusion_strategies[strategy],
            )
        raise ValueError(f"Unsupported evaluation strategy: '{strategy}'")

    async def evaluate_strategy(
        self,
        queries: list[BenchmarkQuery],
        strategy: str,
        k: int = 10,
    ) -> StrategyEvaluationReport:
        """Run all benchmark queries through the specified strategy and aggregate metrics."""
        if not queries:
            return StrategyEvaluationReport(
                strategy=strategy,
                k=k,
                num_queries=0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_mrr=0.0,
                mean_ndcg_at_k=0.0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                query_results=[],
            )

        results: list[QueryEvaluationResult] = []
        for bq in queries:
            request = SearchRequest(**bq.model_dump(), limit=k)
            response = await self._execute_strategy(
                request=request,
                strategy=strategy,
            )
            results.append(self.evaluate_single_query(bq, response, k=k))

        n = len(results)
        latencies = [r.took_ms for r in results]

        return StrategyEvaluationReport(
            strategy=strategy,
            k=k,
            num_queries=n,
            mean_precision_at_k=round(sum(r.precision_at_k for r in results) / n, 4),
            mean_recall_at_k=round(sum(r.recall_at_k for r in results) / n, 4),
            mean_mrr=round(sum(r.mrr for r in results) / n, 4),
            mean_ndcg_at_k=round(sum(r.ndcg_at_k for r in results) / n, 4),
            mean_latency_ms=round(sum(latencies) / n, 2),
            p50_latency_ms=self.metrics.percentile(latencies, 50.0),
            p95_latency_ms=self.metrics.percentile(latencies, 95.0),
            query_results=results,
        )

    async def evaluate_all(
        self,
        queries: list[BenchmarkQuery],
        k: int = 10,
        strategies: tuple[str, ...] | None = None,
    ) -> list[StrategyEvaluationReport]:
        """Evaluate all configured strategies sequentially and return their reports."""
        active_strategies = strategies or ("bm25", "knn", *self.fusion_strategies.keys())
        reports: list[StrategyEvaluationReport] = []
        for strategy in active_strategies:
            report = await self.evaluate_strategy(queries=queries, strategy=strategy, k=k)
            reports.append(report)
        return reports

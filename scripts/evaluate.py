"""CLI script to run Information Retrieval benchmark evaluation across search strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.es_client import (
    check_elasticsearch_health,
    close_elasticsearch_client,
    get_elasticsearch_client,
)
from src.repositories.jobs_repository import ElasticsearchJobRepository
from src.schemas.evaluation import BenchmarkQuery, StrategyEvaluationReport
from src.services.embedder import JobEmbedder
from src.services.evaluation import SearchEvaluator
from src.services.search_service import SearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def load_ground_truth(path: Path) -> list[BenchmarkQuery]:
    """Load and validate ground-truth benchmark queries from a JSON file."""
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuery.model_validate(item) for item in raw_data]


def format_markdown_table(reports: list[StrategyEvaluationReport], k: int) -> str:
    """Format strategy evaluation reports as a clean Markdown comparison table."""
    lines = [
        f"| Strategy | NDCG@{k} | MRR | Recall@{k} | Precision@{k} | Mean (ms) | p50 (ms) | p95 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in reports:
        lines.append(
            f"| **{r.strategy}** "
            f"| {r.mean_ndcg_at_k:.4f} "
            f"| {r.mean_mrr:.4f} "
            f"| {r.mean_recall_at_k:.4f} "
            f"| {r.mean_precision_at_k:.4f} "
            f"| {r.mean_latency_ms:.2f} "
            f"| {r.p50_latency_ms:.2f} "
            f"| {r.p95_latency_ms:.2f} |"
        )
    return "\n".join(lines)


async def run_evaluation(
    ground_truth_path: Path,
    output_path: Path,
    k: int = 10,
) -> int:
    """Execute evaluation benchmark against Elasticsearch and save results."""
    if not ground_truth_path.exists():
        logger.error("Ground truth file not found: %s", ground_truth_path)
        return 1

    queries = load_ground_truth(ground_truth_path)
    logger.info("Loaded %d benchmark queries from %s", len(queries), ground_truth_path)

    client = await get_elasticsearch_client()
    try:
        health = await check_elasticsearch_health(client)
        if not health.get("connected"):
            logger.error("Cannot connect to Elasticsearch: %s", health.get("error"))
            return 1

        embedder = JobEmbedder()
        # Warm up embedding model before timing benchmark queries
        logger.info("Warming up embedding model '%s'...", embedder.model_name)
        embedder.embed_query("warmup query")

        repo = ElasticsearchJobRepository(client=client)
        search_service = SearchService(repository=repo, embedder=embedder)
        evaluator = SearchEvaluator(search_service=search_service)

        logger.info("Evaluating strategies (k=%d) across %d queries...", k, len(queries))
        reports = await evaluator.evaluate_all(queries=queries, k=k)

        table = format_markdown_table(reports, k=k)
        print("\n" + "=" * 86)
        print(f"IR-ARENA BENCHMARK EVALUATION REPORT (k={k}, queries={len(queries)})")
        print("=" * 86)
        print(table)
        print("=" * 86 + "\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [r.model_dump() for r in reports]
        output_path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved detailed evaluation report to %s", output_path)
        return 0
    finally:
        await close_elasticsearch_client()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IR-Arena search quality benchmark across BM25, kNN, RRF, and Weighted Fusion."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=PROJECT_ROOT / "data" / "ground_truth.json",
        help="Path to ground_truth.json benchmark file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation_results.json",
        help="Path to save evaluation_results.json report",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cut-off rank k for NDCG@k, Recall@k, and Precision@k (default: 10)",
    )
    args = parser.parse_args()
    code = asyncio.run(
        run_evaluation(
            ground_truth_path=args.ground_truth,
            output_path=args.output,
            k=args.k,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

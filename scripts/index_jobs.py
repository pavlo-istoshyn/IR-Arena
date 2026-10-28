"""CLI script to generate embeddings and index jobs into Elasticsearch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.es_client import (
    check_elasticsearch_health,
    close_elasticsearch_client,
    get_elasticsearch_client,
)
from src.repositories import JOBS_INDEX_NAME, JobsIndexManager
from src.schemas.job import JobListing
from src.services.embedder import JobEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index_jobs")


async def run_indexing(
    file_path: Path,
    recreate: bool = False,
    batch_size: int = 64,
    limit: int | None = None,
) -> int:
    """Read jobs from JSONL, generate embeddings, and index into Elasticsearch."""
    if not file_path.exists():
        logger.error("Dataset file not found: %s", file_path)
        return 1

    logger.info("Reading jobs from: %s", file_path)
    jobs: list[JobListing] = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            stripped = line.strip()
            if stripped:
                try:
                    jobs.append(JobListing.model_validate_json(stripped))
                except Exception as exc:
                    logger.warning("Skipping malformed line %d: %s", idx + 1, exc)

    logger.info("Loaded %d valid job listings.", len(jobs))
    if not jobs:
        logger.warning("No jobs to index.")
        return 0

    client = await get_elasticsearch_client()
    try:
        health = await check_elasticsearch_health(client)
        if not health.get("connected"):
            logger.error("Cannot connect to Elasticsearch at %s: %s", client, health.get("error"))
            logger.error("Please make sure Docker is running: `docker compose up -d`")
            return 1

        logger.info(
            "Connected to Elasticsearch cluster '%s' (version %s, status %s).",
            health.get("cluster_name"),
            health.get("version"),
            health.get("cluster_status"),
        )

        index_manager = JobsIndexManager(client=client)
        await index_manager.create_index(recreate=recreate)

        embedder = JobEmbedder()
        logger.info(
            "Generating embeddings and indexing in batches of %d (Model: %s, Dims: %d)...",
            batch_size,
            embedder.model_name,
            embedder.dimension,
        )

        total_indexed = 0
        total_errors = 0
        start_time = time.perf_counter()

        for start_idx in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start_idx : start_idx + batch_size]
            b_start = time.perf_counter()

            embeddings = embedder.embed_jobs(batch_jobs, batch_size=batch_size)
            success, errs = await index_manager.index_jobs_bulk(batch_jobs, embeddings)
            total_indexed += success
            total_errors += errs

            elapsed_b = time.perf_counter() - b_start
            logger.info(
                "Indexed %d / %d jobs (+%d in %.2fs) | Total errors: %d",
                total_indexed,
                len(jobs),
                len(batch_jobs),
                elapsed_b,
                total_errors,
            )

        total_elapsed = time.perf_counter() - start_time

        # Refresh index so search is immediately visible
        await client.indices.refresh(index=JOBS_INDEX_NAME)
        count_res = await client.count(index=JOBS_INDEX_NAME)
        doc_count = count_res.get("count", 0)

        logger.info("=" * 60)
        logger.info("INDEXING COMPLETED SUCCESSFULLY")
        logger.info("Total documents in index '%s': %d", JOBS_INDEX_NAME, doc_count)
        logger.info(
            "Total time: %.2f seconds (%.1f docs/sec)",
            total_elapsed,
            total_indexed / max(total_elapsed, 0.01),
        )
        logger.info("=" * 60)
        return 0
    finally:
        await close_elasticsearch_client()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index jobs into Elasticsearch with dense vectors."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=PROJECT_ROOT / "data" / "jobs.jsonl",
        help="Path to jobs.jsonl file (default: data/jobs.jsonl)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Elasticsearch index if it already exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding and bulk indexing (default: 64)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of jobs to index (useful for fast testing)",
    )

    args = parser.parse_args()
    code = asyncio.run(
        run_indexing(
            file_path=args.file,
            recreate=args.recreate,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

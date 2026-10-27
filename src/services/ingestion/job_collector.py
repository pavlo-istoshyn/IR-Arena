"""Service class for orchestrating job collection and pagination from Arbeitsagentur."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from src.clients.arbeitsagentur import ArbeitsagenturClient
from src.schemas.job import JobListing
from src.services.ingestion.job_filter import JobRoleFilter
from src.services.ingestion.job_normalizer import JobNormalizer

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_QUERIES = [
    "Python",
    "Python Backend",
    "Softwareentwickler Python",
    "FastAPI",
    "Backend Entwickler Python",
    "Junior Python",
    "Python AI",
    "AI Engineer Python",
    "Machine Learning Python",
    "LLM Python",
    "Python Developer",
    "Backend Developer Python",
    "Python REST API",
    "Django Python",
    "Data Engineer Python",
]


class JobCollector:
    """Orchestrates job search pagination, deduplication, detail fetching, and JSONL export."""

    def __init__(
        self,
        client: ArbeitsagenturClient | None = None,
        role_filter: JobRoleFilter | None = None,
        normalizer: JobNormalizer | None = None,
    ) -> None:
        self.client = client
        self.role_filter = role_filter or JobRoleFilter()
        self.normalizer = normalizer or JobNormalizer()

    async def _gather_candidates(
        self,
        client: ArbeitsagenturClient,
        queries: list[str],
        target_limit: int,
        berufsfeld: str | None,
        test_mode: bool,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Paginate across target search queries to gather candidate job references."""
        start_time = time.time()
        max_page = 2 if test_mode else 25
        active_queries = list(queries)
        seen_refnrs: set[str] = set()
        collected_items: list[tuple[str, dict[str, Any]]] = []

        for page in range(1, max_page + 1):
            if len(seen_refnrs) >= target_limit or not active_queries:
                break

            exhausted: list[str] = []
            for query in active_queries:
                if len(seen_refnrs) >= target_limit:
                    break

                items, max_avail = await client.search_jobs_page(
                    query=query,
                    page=page,
                    size=50,
                    berufsfeld=berufsfeld,
                )
                if not items or (page * 50) >= max_avail:
                    exhausted.append(query)

                new_in_page = 0
                for item in items:
                    refnr = item.get("referenznummer")
                    title = item.get("stellenangebotsTitel") or item.get("titel") or ""
                    if self.role_filter.is_non_target_role(title):
                        continue
                    if refnr and refnr not in seen_refnrs:
                        seen_refnrs.add(refnr)
                        collected_items.append((refnr, item))
                        new_in_page += 1
                        if len(seen_refnrs) >= target_limit:
                            break

                logger.info(
                    "Query '%s' (page %d): +%d unique jobs (total: %d / %d)",
                    query,
                    page,
                    new_in_page,
                    len(seen_refnrs),
                    target_limit,
                )

            for q in exhausted:
                if q in active_queries:
                    active_queries.remove(q)

        duration = time.time() - start_time
        logger.info("Gathered %d unique job listings in %.2fs.", len(collected_items), duration)
        return collected_items

    async def _fetch_and_save_details(
        self,
        client: ArbeitsagenturClient,
        candidates: list[tuple[str, dict[str, Any]]],
        output_path: Path,
        concurrency: int,
    ) -> list[JobListing]:
        """Fetch full details concurrently for candidate jobs and stream them to JSONL."""
        semaphore = asyncio.Semaphore(concurrency)
        processed_jobs: list[JobListing] = []
        batch_size = 25
        total_to_fetch = len(candidates)

        with open(output_path, "w", encoding="utf-8") as out_file:
            for i in range(0, total_to_fetch, batch_size):
                batch = candidates[i : i + batch_size]
                tasks = [client.fetch_job_details(refnr, semaphore) for refnr, _ in batch]
                details_results = await asyncio.gather(*tasks)

                for (refnr, search_item), details in zip(batch, details_results, strict=True):
                    try:
                        job_listing = self.normalizer.normalize(refnr, search_item, details)
                        processed_jobs.append(job_listing)
                        out_file.write(job_listing.model_dump_json() + "\n")
                    except Exception as err:
                        logger.error("Failed to parse job %s: %s", refnr, err)

                logger.info("Fetched details: %d / %d jobs", len(processed_jobs), total_to_fetch)

        return processed_jobs

    async def collect(
        self,
        queries: list[str],
        output_path: Path,
        limit: int = 1500,
        concurrency: int = 5,
        berufsfeld: str | None = "Informatik",
        test_mode: bool = False,
    ) -> list[JobListing]:
        """
        Collect jobs from Arbeitsagentur API using round-robin pagination,
        deduplicate by refnr, fetch full details concurrently, and write to JSONL.
        """
        target_limit = min(limit, 20 if test_mode else limit)
        logger.info(
            "Starting job collection. Target: %d jobs. Concurrency: %d", target_limit, concurrency
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cm = nullcontext(self.client) if self.client is not None else ArbeitsagenturClient()
        async with cm as active_client:
            candidates = await self._gather_candidates(
                client=active_client,
                queries=queries,
                target_limit=target_limit,
                berufsfeld=berufsfeld,
                test_mode=test_mode,
            )
            return await self._fetch_and_save_details(
                client=active_client,
                candidates=candidates,
                output_path=output_path,
                concurrency=concurrency,
            )

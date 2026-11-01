"""Script to re-extract skills and detect seniority for all jobs in data/jobs.jsonl and Elasticsearch."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elasticsearch.helpers import async_bulk

from src.core.es_client import (
    check_elasticsearch_health,
    close_elasticsearch_client,
    get_elasticsearch_client,
)
from src.repositories.base import JOBS_INDEX_NAME
from src.schemas.job import JobListing
from src.services.ingestion.seniority import detect_seniority
from src.services.ingestion.skill_extractor import SkillExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_jobs")


async def enrich_and_update(file_path: Path = PROJECT_ROOT / "data" / "jobs.jsonl") -> None:
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return

    logger.info("Reading jobs from %s...", file_path)
    extractor = SkillExtractor()

    updated_jobs: list[JobListing] = []
    seniority_counts: Counter[str] = Counter()
    all_skills_counter: Counter[str] = Counter()

    with open(file_path, encoding="utf-8") as f:
        for _idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            raw_data = json.loads(line_str)
            title = raw_data.get("title", "")
            desc = raw_data.get("description", "")

            # Re-extract skills using expanded 100+ patterns
            combined_text = f"{title}\n{desc}"
            skills = extractor.extract_skills(combined_text)
            seniority = detect_seniority(title=title, description=desc)

            raw_data["skills"] = skills
            raw_data["seniority"] = seniority

            job = JobListing.model_validate(raw_data)
            updated_jobs.append(job)

            seniority_counts[seniority] += 1
            for sk in skills:
                all_skills_counter[sk] += 1

    logger.info("Loaded and enriched %d job listings.", len(updated_jobs))
    logger.info("Seniority breakdown: %s", dict(seniority_counts))
    logger.info("Top 20 detected skills: %s", all_skills_counter.most_common(20))

    # Write back to data/jobs.jsonl atomically
    temp_path = file_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for job in updated_jobs:
            f.write(job.model_dump_json() + "\n")
    temp_path.replace(file_path)
    logger.info("Successfully updated %s", file_path)

    # Update Elasticsearch if connected
    client = await get_elasticsearch_client()
    try:
        health = await check_elasticsearch_health(client)
        if not health.get("connected"):
            logger.warning("Elasticsearch not connected. Skipping ES update.")
            return

        exists = bool(await client.indices.exists(index=JOBS_INDEX_NAME))
        if not exists:
            logger.warning("Index '%s' does not exist in ES. Skipping bulk update.", JOBS_INDEX_NAME)
            return

        # Ensure mapping includes seniority
        logger.info("Updating Elasticsearch mapping for index '%s'...", JOBS_INDEX_NAME)
        await client.indices.put_mapping(
            index=JOBS_INDEX_NAME,
            properties={"seniority": {"type": "keyword"}},
        )

        logger.info("Sending bulk update to Elasticsearch for %d documents...", len(updated_jobs))
        actions = [
            {
                "_op_type": "update",
                "_index": JOBS_INDEX_NAME,
                "_id": job.id,
                "doc": {
                    "skills": job.skills,
                    "seniority": job.seniority,
                },
            }
            for job in updated_jobs
        ]

        success, errors = await async_bulk(client, actions, raise_on_error=False)
        error_count = len(errors) if isinstance(errors, list) else 0
        logger.info("ES bulk update done: %d succeeded, %d errors.", success, error_count)

        await client.indices.refresh(index=JOBS_INDEX_NAME)
        logger.info("Refreshed index '%s'. All updates are live!", JOBS_INDEX_NAME)
    finally:
        await close_elasticsearch_client()


if __name__ == "__main__":
    asyncio.run(enrich_and_update())

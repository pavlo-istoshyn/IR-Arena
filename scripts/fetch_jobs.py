"""Thin CLI script for collecting jobs via the JobCollector service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.ingestion import DEFAULT_PROFILE_QUERIES, JobCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape and extract Arbeitsagentur jobs matching Python/AI stack."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Target number of unique jobs to collect (default: 1500).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Fast test mode: collects ~20 jobs to verify pipeline functionality.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent HTTP requests for job details (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "jobs.jsonl"),
        help="Output path for JSONL file (default: data/jobs.jsonl).",
    )
    parser.add_argument(
        "--berufsfeld",
        type=str,
        default="Informatik",
        help="Arbeitsagentur berufsfeld filter (default: 'Informatik').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    berufsfeld = args.berufsfeld if args.berufsfeld.strip() else None
    output_path = Path(args.output).resolve()

    collector = JobCollector()
    asyncio.run(
        collector.collect(
            queries=DEFAULT_PROFILE_QUERIES,
            output_path=output_path,
            limit=args.limit,
            concurrency=args.concurrency,
            berufsfeld=berufsfeld,
            test_mode=args.test,
        )
    )


if __name__ == "__main__":
    main()

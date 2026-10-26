"""Async Elasticsearch client management and health checks."""

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: AsyncElasticsearch | None = None


def create_elasticsearch_client(settings: Settings | None = None) -> AsyncElasticsearch:
    """Create a new AsyncElasticsearch client instance."""
    cfg = (settings or get_settings()).elasticsearch
    return AsyncElasticsearch(
        hosts=[cfg.url],
        request_timeout=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
        retry_on_timeout=True,
    )


async def get_elasticsearch_client() -> AsyncElasticsearch:
    """Dependency / accessor for the global AsyncElasticsearch client."""
    global _client
    if _client is None:
        _client = create_elasticsearch_client()
    return _client


async def close_elasticsearch_client() -> None:
    """Close the global AsyncElasticsearch client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def check_elasticsearch_health(
    client: AsyncElasticsearch | None = None,
) -> dict[str, Any]:
    """Perform a health check against the Elasticsearch cluster."""
    es = client or await get_elasticsearch_client()
    try:
        is_reachable = await es.ping()
        if not is_reachable:
            return {
                "status": "unhealthy",
                "connected": False,
                "message": "Elasticsearch host responded with ping failure.",
            }

        info = await es.info()
        health = await es.cluster.health()

        return {
            "status": "healthy",
            "connected": True,
            "cluster_name": health.get("cluster_name"),
            "cluster_status": health.get("status"),
            "number_of_nodes": health.get("number_of_nodes"),
            "version": info.get("version", {}).get("number"),
            "lucene_version": info.get("version", {}).get("lucene_version"),
        }
    except Exception as exc:
        logger.warning("Elasticsearch health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(exc),
            "message": "Could not connect to Elasticsearch. Is Docker container running?",
        }

"""System and health check router."""

from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from src.core.config import Settings, get_settings
from src.core.es_client import check_elasticsearch_health

router = APIRouter()


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Service is online"},
    },
    tags=["General"],
)
async def root(cfg: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Root endpoint returning basic service info."""
    return {
        "service": cfg.app.app_name,
        "status": "online",
        "environment": cfg.app.app_env,
        "docs_url": "/docs",
    }


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Application and Elasticsearch cluster are healthy"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Elasticsearch cluster is unreachable or degraded"
        },
    },
    tags=["Monitoring"],
)
async def health_check() -> JSONResponse:
    """Health check endpoint returning application and Elasticsearch status."""
    es_health = await check_elasticsearch_health()

    payload = {
        "status": "ok" if es_health.get("connected") else "degraded",
        "elasticsearch": es_health,
    }

    http_status = (
        status.HTTP_200_OK if es_health.get("connected") else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(status_code=http_status, content=payload)


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Platform metadata and current configuration"},
    },
    tags=["General"],
)
async def app_info(cfg: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return platform metadata and current configuration."""
    return {
        "app_name": cfg.app.app_name,
        "version": cfg.app.version,
        "environment": cfg.app.app_env,
        "debug": cfg.app.debug,
        "elasticsearch_url": cfg.elasticsearch.url,
        "default_embedding_model": cfg.ml.default_embedding_model,
    }

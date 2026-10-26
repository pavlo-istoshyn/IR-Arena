"""Core module for IR-Arena: configuration, exceptions, and infrastructure clients."""

from src.core.config import (
    AppConfig,
    ElasticsearchConfig,
    MLConfig,
    Settings,
    get_settings,
)
from src.core.es_client import (
    check_elasticsearch_health,
    close_elasticsearch_client,
    create_elasticsearch_client,
    get_elasticsearch_client,
)
from src.core.exceptions import IRArenaError, TransientAPIError

__all__ = [
    "AppConfig",
    "ElasticsearchConfig",
    "MLConfig",
    "Settings",
    "get_settings",
    "check_elasticsearch_health",
    "close_elasticsearch_client",
    "create_elasticsearch_client",
    "get_elasticsearch_client",
    "IRArenaError",
    "TransientAPIError",
]

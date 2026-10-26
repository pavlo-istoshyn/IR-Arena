"""Application configuration using layered Pydantic v2 Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration shared across settings classes."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class ElasticsearchConfig(BaseConfig):
    """Elasticsearch connection and client configuration."""

    model_config = BaseConfig.model_config | {"env_prefix": "ELASTICSEARCH_"}

    url: str = "http://localhost:9200"
    timeout_seconds: float = 30.0
    max_retries: int = 3


class AppConfig(BaseConfig):
    """Application and server runtime configuration."""

    app_name: str = "IR-Arena"
    version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000


class MLConfig(BaseConfig):
    """Machine learning and embedding model configuration."""

    default_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384


class ArbeitsagenturConfig(BaseConfig):
    """Bundesagentur für Arbeit API configuration."""

    model_config = BaseConfig.model_config | {"env_prefix": "ARBEITSAGENTUR_"}

    base_url: str = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4"
    api_key: str = "jobboerse-jobsuche"
    user_agent: str = "IR-Arena/1.0"
    timeout_seconds: float = 15.0
    max_retries: int = 3


class Settings(BaseConfig):
    """Master application configuration composing sub-configs."""

    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    arbeitsagentur: ArbeitsagenturConfig = Field(default_factory=ArbeitsagenturConfig)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()

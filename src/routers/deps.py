"""FastAPI dependencies for search services and repositories."""

from elasticsearch import AsyncElasticsearch
from fastapi import Depends

from src.core.es_client import get_elasticsearch_client
from src.repositories.jobs_repository import ElasticsearchJobRepository
from src.services.embedder import JobEmbedder, get_embedder
from src.services.search_service import SearchService


async def get_search_service(
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client),
    embedder: JobEmbedder = Depends(get_embedder),
) -> SearchService:
    """Dependency provider injecting ElasticsearchJobRepository into a stateless SearchService."""
    repo = ElasticsearchJobRepository(client=es_client)
    return SearchService(repository=repo, embedder=embedder)

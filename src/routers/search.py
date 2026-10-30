"""Search API endpoints for BM25, kNN vector, and Hybrid RRF retrieval."""

from fastapi import APIRouter, Depends, status

from src.routers.deps import get_search_service
from src.schemas.search import SearchRequest, SearchResponse
from src.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post(
    "/bm25",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="BM25 Search",
    description="BM25 lexical search with German analyzer, title/skills boosting, and typo tolerance.",
)
async def search_bm25_endpoint(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute BM25 text search."""
    return await search_service.search_bm25(request)


@router.post(
    "/knn",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="kNN Vector Search",
    description="Dense vector search using multilingual SentenceTransformers embeddings and cosine similarity.",
)
async def search_knn_endpoint(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute semantic kNN vector search."""
    return await search_service.search_knn(request)


@router.post(
    "/hybrid",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid Search (RRF)",
    description="Runs BM25 and kNN concurrently, fusing results via Reciprocal Rank Fusion.",
)
async def search_hybrid_endpoint(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Execute hybrid search with Reciprocal Rank Fusion."""
    return await search_service.search_hybrid(request)

"""FastAPI application entrypoint for IR-Arena."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.core.es_client import close_elasticsearch_client, get_elasticsearch_client
from src.routers import search_router, system_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    _ = await get_elasticsearch_client()
    yield
    await close_elasticsearch_client()


settings = get_settings()

app = FastAPI(
    title=settings.app.app_name,
    description="IR-Arena: Hybrid Search & Retrieval Evaluation Platform",
    version=settings.app.version,
    debug=settings.app.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(search_router)

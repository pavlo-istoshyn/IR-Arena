"""Async HTTP Client for the Bundesagentur für Arbeit Jobsuche REST API."""

import asyncio
import base64
import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_incrementing,
)

from src.core.config import ArbeitsagenturConfig
from src.core.exceptions import TransientAPIError

logger = logging.getLogger(__name__)


class ArbeitsagenturClient:
    """Async HTTP client for the Bundesagentur für Arbeit public job search API."""

    RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

    def __init__(
        self,
        config: ArbeitsagenturConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        cfg = config or ArbeitsagenturConfig()
        self.base_url = cfg.base_url.rstrip("/")
        self.headers = {
            "X-API-Key": cfg.api_key,
            "User-Agent": cfg.user_agent,
        }
        self.timeout = cfg.timeout_seconds
        self.max_retries = cfg.max_retries
        self._client: httpx.AsyncClient | None = http_client
        self._owns_client: bool = http_client is None

    async def __aenter__(self) -> "ArbeitsagenturClient":
        """Initialize and open connection pool if not provided externally."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout,
            )
            self._owns_client = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close connection pool if managed internally."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Return active client or raise RuntimeError if session is not active."""
        if self._client is None:
            raise RuntimeError(
                "ArbeitsagenturClient session is not active. "
                "Use 'async with ArbeitsagenturClient() as client:' or pass an initialized http_client to __init__."
            )
        return self._client

    async def _request_with_retry(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> httpx.Response | None:
        """Execute GET request with Tenacity retry on transient HTTP and connection errors."""
        async def _make_request() -> httpx.Response | None:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_incrementing(start=0.5, increment=0.5),
                retry=retry_if_exception_type(
                    (httpx.RequestError, httpx.TimeoutException, TransientAPIError)
                ),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                retry_error_callback=lambda _state: None,
            )
            async for attempt in retrying:
                with attempt:
                    response = await self.client.get(url, params=params)
                    if response.status_code == 200:
                        return response
                    if response.status_code in (404, 410):
                        return None
                    if response.status_code in self.RETRY_STATUS_CODES:
                        raise TransientAPIError(
                            f"Transient HTTP {response.status_code} received from {url}"
                        )
                    return response
            return None

        if semaphore:
            async with semaphore:
                return await _make_request()
        return await _make_request()

    async def search_jobs_page(
        self,
        query: str,
        page: int = 1,
        size: int = 50,
        berufsfeld: str | None = "Informatik",
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Query job listings for a specific text query and page number.

        Returns:
            tuple of (list of raw job dicts, total matching count)
        """
        url = f"{self.base_url}/jobs"
        params: dict[str, Any] = {
            "was": query,
            "page": page,
            "size": size,
        }
        if berufsfeld:
            params["berufsfeld"] = berufsfeld

        response = await self._request_with_retry(url, params=params)
        if response is None:
            return [], 0
        data = response.json()
        return data.get("stellenangebote", []), data.get("maxErgebnisse", 0)

    async def fetch_job_details(
        self,
        refnr: str,
        semaphore: asyncio.Semaphore | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch full job details for a given reference number (refnr).
        The API requires base64-encoding the reference number.
        """
        encoded_refnr = base64.b64encode(refnr.encode("utf-8")).decode("utf-8")
        url = f"{self.base_url}/jobdetails/{encoded_refnr}"
        response = await self._request_with_retry(url, semaphore=semaphore)
        return response.json() if response is not None else None

"""Unit tests for ArbeitsagenturClient."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.clients.arbeitsagentur import ArbeitsagenturClient
from src.core.config import ArbeitsagenturConfig


@pytest.mark.asyncio
async def test_client_init_and_context_manager():
    config = ArbeitsagenturConfig(
        base_url="https://test.api.com",
        api_key="test-key",
        timeout_seconds=5.0,
    )
    client = ArbeitsagenturClient(config=config)
    assert client.base_url == "https://test.api.com"
    assert client.headers["X-API-Key"] == "test-key"

    async with client as active_client:
        assert active_client._client is not None
        assert isinstance(active_client._client, httpx.AsyncClient)

    assert client._client is None
    with pytest.raises(RuntimeError, match="session is not active"):
        _ = client.client


@pytest.mark.asyncio
async def test_client_search_jobs_success():
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "stellenangebote": [{"referenznummer": "123", "titel": "Python Dev"}],
        "maxErgebnisse": 100,
    }
    mock_http_client.get.return_value = mock_response

    client = ArbeitsagenturClient(http_client=mock_http_client)

    items, total = await client.search_jobs_page("Python", page=1, size=10)
    assert len(items) == 1
    assert items[0]["referenznummer"] == "123"
    assert total == 100


@pytest.mark.asyncio
async def test_client_fetch_details_success():
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"beruf": "Softwareentwickler"}
    mock_http_client.get.return_value = mock_response

    client = ArbeitsagenturClient(http_client=mock_http_client)

    details = await client.fetch_job_details("10000-12345")
    assert details is not None
    assert details["beruf"] == "Softwareentwickler"


@pytest.mark.asyncio
async def test_client_request_with_retry_404():
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_http_client.get.return_value = mock_resp

    client = ArbeitsagenturClient(http_client=mock_http_client)

    resp = await client._request_with_retry("https://test.api.com/nonexistent")
    assert resp is None


@pytest.mark.asyncio
async def test_client_request_retry_on_500_then_success():
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    resp_500 = MagicMock(status_code=500)
    resp_200 = MagicMock(status_code=200)
    resp_200.json.return_value = {"status": "ok"}
    mock_http_client.get.side_effect = [resp_500, resp_200]

    config = ArbeitsagenturConfig(max_retries=2)
    client = ArbeitsagenturClient(config=config, http_client=mock_http_client)

    resp = await client._request_with_retry("https://test.api.com/retry-endpoint")
    assert resp is not None
    assert resp.status_code == 200
    assert mock_http_client.get.call_count == 2


@pytest.mark.asyncio
async def test_client_request_retry_exhausted_returns_none():
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    resp_503 = MagicMock(status_code=503)
    mock_http_client.get.return_value = resp_503

    config = ArbeitsagenturConfig(max_retries=2)
    client = ArbeitsagenturClient(config=config, http_client=mock_http_client)

    resp = await client._request_with_retry("https://test.api.com/flaky-endpoint")
    assert resp is None
    assert mock_http_client.get.call_count == 2

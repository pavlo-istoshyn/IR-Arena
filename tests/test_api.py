from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.config import AppConfig, ElasticsearchConfig, MLConfig, Settings, get_settings
from src.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "IR-Arena"
    assert data["status"] == "online"
    assert data["docs_url"] == "/docs"


@pytest.mark.asyncio
async def test_info_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "IR-Arena"
    assert "elasticsearch_url" in data
    assert "default_embedding_model" in data


@pytest.mark.asyncio
async def test_health_endpoint_healthy():
    mock_health = {
        "status": "healthy",
        "connected": True,
        "cluster_name": "ir-arena-cluster",
        "cluster_status": "green",
        "number_of_nodes": 1,
        "version": "8.15.0",
    }
    with patch(
        "src.routers.system.check_elasticsearch_health", new=AsyncMock(return_value=mock_health)
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["elasticsearch"]["connected"] is True


@pytest.mark.asyncio
async def test_health_endpoint_unhealthy():
    mock_health = {
        "status": "unhealthy",
        "connected": False,
        "message": "Elasticsearch host responded with ping failure.",
    }
    with patch(
        "src.routers.system.check_elasticsearch_health", new=AsyncMock(return_value=mock_health)
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["elasticsearch"]["connected"] is False


@pytest.mark.asyncio
async def test_endpoints_with_custom_settings_dependency():
    custom_settings = Settings(
        app=AppConfig(app_name="Custom-Service", app_env="test-env", debug=False),
        elasticsearch=ElasticsearchConfig(url="http://custom-api-es:9200"),
        ml=MLConfig(default_embedding_model="custom-api-bert"),
    )
    app.dependency_overrides[get_settings] = lambda: custom_settings
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res_root = await client.get("/")
            assert res_root.status_code == 200
            root_data = res_root.json()
            assert root_data["service"] == "Custom-Service"
            assert root_data["environment"] == "test-env"

            res_info = await client.get("/info")
            assert res_info.status_code == 200
            info_data = res_info.json()
            assert info_data["app_name"] == "Custom-Service"
            assert info_data["environment"] == "test-env"
            assert info_data["debug"] is False
            assert info_data["elasticsearch_url"] == "http://custom-api-es:9200"
            assert info_data["default_embedding_model"] == "custom-api-bert"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cors_headers_present():
    transport = ASGITransport(app=app)
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    }
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options("/search/bm25", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

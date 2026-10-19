from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from src.core.config import get_settings
from src.core.dependencies import get_document_service, get_rag_pipeline_service
from src.core.exceptions import DocumentProcessingError, VectorStoreError
from src.domains.documents.schemas import DocumentUploadResponse
from src.domains.rag.schemas import QueryResponse, SearchResultChunk
from src.main import app


def auth_headers():
    token = get_settings().API_AUTH_TOKEN
    return {"Authorization": f"Bearer {token.get_secret_value()}"} if token else {}


@pytest.mark.asyncio
async def test_health_check_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app_env": get_settings().APP_ENV}


@pytest.mark.asyncio
async def test_upload_endpoint_with_dependency_override():
    mock_doc_service = AsyncMock()
    mock_doc_service.process_and_index_document.return_value = DocumentUploadResponse(
        document_id="test-doc-id",
        filename="test.txt",
        total_chunks=3,
    )

    app.dependency_overrides[get_document_service] = lambda: mock_doc_service

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("test.txt", b"hello world", "text/plain")}
            response = await ac.post("/api/v1/documents/upload", files=files, headers=auth_headers())
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "test-doc-id"
        assert data["total_chunks"] == 3
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_rag_query_endpoint_with_dependency_override():
    mock_pipeline_service = AsyncMock()
    mock_pipeline_service.execute_query.return_value = QueryResponse(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        source_chunks=[
            SearchResultChunk(
                chunk_id="c1",
                document_id="d1",
                content="RAG overview",
                score=0.95,
            )
        ],
    )

    app.dependency_overrides[get_rag_pipeline_service] = lambda: mock_pipeline_service

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/rag/query",
                json={"query": "What is RAG?"},
                headers=auth_headers(),
            )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is RAG?"
        assert "Retrieval-Augmented Generation" in data["answer"]
        assert len(data["source_chunks"]) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/documents/upload",
            files={"file": ("payload.exe", b"not executable", "application/octet-stream")},
            headers=auth_headers(),
        )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_query_rejects_invalid_limits():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/rag/query",
            json={"query": "hello", "top_k": 0},
            headers=auth_headers(),
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_routes_reject_missing_bearer_token(monkeypatch):
    from src.core import security

    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(API_AUTH_TOKEN=SecretStr("expected-token"), APP_ENV="development"),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/rag/query", json={"query": "hello"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_upload_enforces_configured_size_limit(monkeypatch):
    import src.domains.documents.controller as document_controller

    monkeypatch.setattr(
        document_controller,
        "get_settings",
        lambda: SimpleNamespace(MAX_UPLOAD_SIZE_BYTES=5),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/documents/upload",
            files={"file": ("large.txt", b"123456", "text/plain")},
            headers=auth_headers(),
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_document_processing_error_maps_to_client_error():
    mock_doc_service = AsyncMock()
    mock_doc_service.process_and_index_document.side_effect = DocumentProcessingError("invalid document")
    app.dependency_overrides[get_document_service] = lambda: mock_doc_service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/documents/upload",
                files={"file": ("broken.txt", b"some text", "text/plain")},
                headers=auth_headers(),
            )
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_vector_store_error_maps_to_service_unavailable():
    mock_pipeline_service = AsyncMock()
    mock_pipeline_service.execute_query.side_effect = VectorStoreError("vector store unavailable")
    app.dependency_overrides[get_rag_pipeline_service] = lambda: mock_pipeline_service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/rag/query",
                json={"query": "hello"},
                headers=auth_headers(),
            )
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()

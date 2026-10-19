from unittest.mock import AsyncMock
import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
from src.core.dependencies import get_document_service, get_rag_pipeline_service
from src.domains.documents.schemas import DocumentUploadResponse
from src.domains.rag.schemas import QueryResponse, SearchResultChunk


@pytest.mark.asyncio
async def test_health_check_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app_env": "development"}


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
            response = await ac.post("/api/v1/documents/upload", files=files)
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
            response = await ac.post("/api/v1/rag/query", json={"query": "What is RAG?"})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is RAG?"
        assert "Retrieval-Augmented Generation" in data["answer"]
        assert len(data["source_chunks"]) == 1
    finally:
        app.dependency_overrides.clear()

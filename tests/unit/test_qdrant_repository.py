import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.domains.documents.schemas import DocumentChunk
from src.infrastructure.qdrant.repository import QdrantVectorRepository
from src.core.exceptions import VectorStoreError


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    # Mock collections check
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock(return_value=True)
    client.upsert = AsyncMock(return_value=True)
    client.search = AsyncMock()
    client.delete = AsyncMock(return_value=True)
    return client


@pytest.fixture
def repository(mock_qdrant_client):
    repo = QdrantVectorRepository(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        api_key=None,
    )
    repo._client = mock_qdrant_client
    return repo


@pytest.mark.asyncio
async def test_ensure_collection_exists_when_already_present(repository, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    await repository.ensure_collection(vector_size=384)
    mock_qdrant_client.collection_exists.assert_awaited_once_with(collection_name="test_collection")
    mock_qdrant_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(repository, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    await repository.ensure_collection(vector_size=384)
    mock_qdrant_client.collection_exists.assert_awaited_once_with(collection_name="test_collection")
    mock_qdrant_client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_chunks_success(repository, mock_qdrant_client):
    chunks = [
        DocumentChunk(
            chunk_id="doc1_0",
            document_id="doc1",
            content="Hello Qdrant world",
            embedding=[0.1, 0.2, 0.3],
            metadata={"filename": "test.txt"},
        ),
        DocumentChunk(
            chunk_id="doc1_1",
            document_id="doc1",
            content="Second chunk text",
            embedding=[0.4, 0.5, 0.6],
            metadata={"filename": "test.txt"},
        ),
    ]

    result = await repository.save_chunks(chunks)
    assert result is True
    mock_qdrant_client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_chunks_empty_list(repository, mock_qdrant_client):
    result = await repository.save_chunks([])
    assert result is True
    mock_qdrant_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_save_chunks_failure_raises_vector_store_error(repository, mock_qdrant_client):
    mock_qdrant_client.upsert.side_effect = Exception("Qdrant connection refused")
    chunks = [
        DocumentChunk(
            chunk_id="doc1_0",
            document_id="doc1",
            content="Fail text",
            embedding=[0.1, 0.2, 0.3],
        )
    ]
    with pytest.raises(VectorStoreError):
        await repository.save_chunks(chunks)


@pytest.mark.asyncio
async def test_search_success(repository, mock_qdrant_client):
    mock_hit = MagicMock()
    mock_hit.id = "doc1_0"
    mock_hit.score = 0.95
    mock_hit.payload = {
        "document_id": "doc1",
        "content": "Found content",
        "filename": "test.txt",
    }
    mock_qdrant_client.search.return_value = [mock_hit]

    query_vector = [0.1, 0.2, 0.3]
    results = await repository.search(query_vector=query_vector, top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "doc1_0"
    assert results[0].document_id == "doc1"
    assert results[0].content == "Found content"
    mock_qdrant_client.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_by_document_id_success(repository, mock_qdrant_client):
    result = await repository.delete_by_document_id("doc1")
    assert result is True
    mock_qdrant_client.delete.assert_awaited_once()

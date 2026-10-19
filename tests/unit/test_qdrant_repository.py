from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qmodels

from src.core.exceptions import VectorStoreError
from src.domains.documents.schemas import DocumentChunk
from src.infrastructure.qdrant.repository import QdrantVectorRepository


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    # Mock collections check
    client.collection_exists = AsyncMock(return_value=True)
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=MagicMock())
            ),
            payload_schema={},
        )
    )
    client.create_collection = AsyncMock(return_value=True)
    client.create_payload_index = AsyncMock(return_value=True)
    client.upsert = AsyncMock(return_value=True)
    client.search = AsyncMock()
    client.delete = AsyncMock(return_value=True)
    return client


@pytest.fixture
def repository(mock_qdrant_client):
    with patch(
        "src.infrastructure.qdrant.repository.AsyncQdrantClient",
        return_value=mock_qdrant_client,
    ):
        repo = QdrantVectorRepository(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            api_key=None,
        )
    return repo


@pytest.mark.asyncio
async def test_ensure_collection_exists_when_already_present(repository, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_qdrant_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
            )
        ),
        payload_schema={},
    )
    await repository.ensure_collection(vector_size=384)
    mock_qdrant_client.collection_exists.assert_awaited_once_with(collection_name="test_collection")
    mock_qdrant_client.create_collection.assert_not_called()
    mock_qdrant_client.create_payload_index.assert_awaited_once_with(
        collection_name="test_collection",
        field_name="filename",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
        wait=True,
    )


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(repository, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    await repository.ensure_collection(vector_size=384)
    mock_qdrant_client.collection_exists.assert_awaited_once_with(collection_name="test_collection")
    mock_qdrant_client.create_collection.assert_awaited_once()
    mock_qdrant_client.create_payload_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_collection_does_not_recreate_existing_filename_index(
    repository,
    mock_qdrant_client,
):
    mock_qdrant_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
            )
        ),
        payload_schema={"filename": qmodels.PayloadSchemaType.KEYWORD},
    )

    await repository.ensure_collection(vector_size=384)

    mock_qdrant_client.create_payload_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_chunks_success(repository, mock_qdrant_client):
    chunks = [
        DocumentChunk(
            chunk_id="doc1_0",
            document_id="doc1",
            content="Hello Qdrant world",
            parent_chunk_id="doc1_parent_0",
            parent_content="Parent passage containing the child text.",
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
    saved_point = mock_qdrant_client.upsert.await_args.kwargs["points"][0]
    assert saved_point.payload["content"] == "Hello Qdrant world"
    assert saved_point.payload["parent_chunk_id"] == "doc1_parent_0"
    assert saved_point.payload["parent_content"] == "Parent passage containing the child text."


@pytest.mark.asyncio
async def test_save_chunks_batches_large_documents(repository, mock_qdrant_client):
    chunks = [
        DocumentChunk(
            chunk_id=f"doc1_{idx}",
            document_id="doc1",
            content=f"chunk {idx}",
            embedding=[0.1, 0.2, 0.3],
        )
        for idx in range(129)
    ]
    await repository.save_chunks(chunks)
    assert mock_qdrant_client.upsert.await_count == 2


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
        "parent_chunk_id": "doc1_parent_0",
        "parent_content": "Parent content with surrounding context.",
    }
    mock_qdrant_client.search.return_value = [mock_hit]

    query_vector = [0.1, 0.2, 0.3]
    results = await repository.search(query_vector=query_vector, top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "doc1_0"
    assert results[0].document_id == "doc1"
    assert results[0].content == "Found content"
    assert results[0].parent_chunk_id == "doc1_parent_0"
    assert results[0].parent_content == "Parent content with surrounding context."
    assert "parent_content" not in results[0].metadata
    mock_qdrant_client.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_by_document_id_success(repository, mock_qdrant_client):
    result = await repository.delete_by_document_id("doc1")
    assert result is True
    mock_qdrant_client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_replace_document_replaces_only_the_matching_document_and_place(
    repository,
    mock_qdrant_client,
):
    existing = MagicMock()
    existing.payload = {
        "document_id": "old-doc",
        "uploaded_at": "2026-10-18T12:00:00+00:00",
    }
    mock_qdrant_client.scroll = AsyncMock(return_value=([existing], None))
    repository.save_chunks = AsyncMock(return_value=True)
    chunks = [
        DocumentChunk(
            chunk_id="new-doc_0",
            document_id="new-doc",
            content="new text",
            metadata={
                "document_identity_key": "contributions",
                "place_key": "camp north",
            },
            embedding=[0.1, 0.2, 0.3],
        )
    ]
    uploaded_at = datetime(2026, 10, 19, tzinfo=timezone.utc)

    result = await repository.replace_document(
        chunks,
        document_identity_key="contributions",
        place_key="camp north",
        uploaded_at=uploaded_at,
    )

    assert result == (True, True)
    repository.save_chunks.assert_awaited_once_with(chunks)
    mock_qdrant_client.delete.assert_awaited_once()
    delete_filter = mock_qdrant_client.delete.await_args.kwargs["points_selector"]
    assert len(delete_filter.must) == 2
    assert len(delete_filter.must_not) == 1


@pytest.mark.asyncio
async def test_replace_document_does_not_overwrite_a_newer_upload(repository, mock_qdrant_client):
    existing = MagicMock()
    existing.payload = {
        "document_id": "newer-doc",
        "uploaded_at": "2026-10-20T12:00:00+00:00",
    }
    mock_qdrant_client.scroll = AsyncMock(return_value=([existing], None))
    repository.save_chunks = AsyncMock(return_value=True)

    result = await repository.replace_document(
        [
            DocumentChunk(
                chunk_id="older-doc_0",
                document_id="older-doc",
                content="old text",
                embedding=[0.1, 0.2, 0.3],
            )
        ],
        document_identity_key="guide",
        place_key="camp north",
        uploaded_at=datetime(2026, 10, 19, tzinfo=timezone.utc),
    )

    assert result == (False, False)
    repository.save_chunks.assert_not_awaited()
    mock_qdrant_client.delete.assert_not_awaited()

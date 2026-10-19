import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domains.documents.repositories import VectorRepository
from src.domains.documents.service import DocumentService
from src.domains.documents.splitters import TextSplitterService
from src.domains.rag.service import EmbeddingService
from src.infrastructure.splitters.langchain_splitter import LangChainRecursiveSplitter


@pytest.fixture
def mock_vector_repo():
    repo = MagicMock(spec=VectorRepository)
    repo.save_chunks = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    service.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return service


@pytest.fixture
def mock_splitter_service():
    splitter = MagicMock(spec=TextSplitterService)
    splitter.split_text = MagicMock(return_value=["chunk 1"])
    return splitter


@pytest.fixture
def document_service(mock_vector_repo, mock_embedding_service, mock_splitter_service):
    return DocumentService(
        vector_repository=mock_vector_repo,
        embedding_service=mock_embedding_service,
        splitter_service=mock_splitter_service,
    )


@pytest.mark.asyncio
async def test_process_and_index_document_calls_repo_with_di(
    document_service,
    mock_vector_repo,
    mock_embedding_service,
    mock_splitter_service,
):
    response = await document_service.process_and_index_document(
        file_content=b"Sample content",
        filename="test.txt",
    )
    assert response.total_chunks == 1
    assert mock_splitter_service.split_text.call_count == 2
    mock_embedding_service.embed_texts.assert_called_once()
    mock_vector_repo.save_chunks.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_document_stores_parent_context_for_each_child(
    document_service,
    mock_vector_repo,
    mock_embedding_service,
    mock_splitter_service,
):
    parent_text = "P" * 521 + "\n\nRelated subparagraph."
    mock_splitter_service.split_text.side_effect = [
        [parent_text],
        ["P" * 500, "P" * 21],
    ]
    mock_embedding_service.embed_texts.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    await document_service.process_and_index_document(
        file_content=parent_text.encode(),
        filename="test.txt",
    )

    saved_chunks = mock_vector_repo.save_chunks.await_args.args[0]
    assert len(saved_chunks) == 2
    assert saved_chunks[0].parent_chunk_id == saved_chunks[1].parent_chunk_id
    assert saved_chunks[0].parent_content == parent_text
    assert saved_chunks[1].parent_content == parent_text
    assert saved_chunks[1].content == "P" * 21
    assert saved_chunks[1].metadata["parent_chunk_index"] == 0
    assert mock_embedding_service.embed_texts.call_args.args[0] == ["P" * 500, "P" * 21]


@pytest.mark.asyncio
async def test_process_document_replaces_same_document_and_place_with_upload_metadata(
    document_service,
    mock_vector_repo,
):
    mock_vector_repo.replace_document = AsyncMock(return_value=(True, True))
    uploaded_at = datetime(2026, 10, 19, 18, 49, tzinfo=timezone.utc)

    response = await document_service.process_and_index_document(
        file_content=b"New contributions",
        filename="contributions.txt",
        document_title="Contributions",
        place="Camp North",
        uploaded_by="Owl",
        uploaded_at=uploaded_at,
    )

    chunks = mock_vector_repo.replace_document.await_args.args[0]
    assert response.replaced_previous is True
    assert chunks[0].metadata["document_title"] == "Contributions"
    assert chunks[0].metadata["place"] == "Camp North"
    assert chunks[0].metadata["uploaded_by"] == "Owl"
    assert chunks[0].metadata["uploaded_at"] == uploaded_at.isoformat()
    assert mock_vector_repo.replace_document.await_args.kwargs == {
        "document_identity_key": "contributions",
        "place_key": "camp north",
        "uploaded_at": uploaded_at,
    }
    mock_vector_repo.save_chunks.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_document_rejects_an_older_replacement(document_service, mock_vector_repo):
    from src.core.exceptions import DocumentProcessingError

    mock_vector_repo.replace_document = AsyncMock(return_value=(False, False))
    with pytest.raises(DocumentProcessingError, match="newer version"):
        await document_service.process_and_index_document(
            file_content=b"Old contributions",
            filename="contributions.txt",
            document_title="Contributions",
            place="Camp North",
            uploaded_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_process_document_parses_file_off_event_loop(document_service):
    event_loop_thread = threading.get_ident()
    parser_thread = []
    document_service.extract_text = lambda *_: (
        parser_thread.append(threading.get_ident()) or "Sample content"
    )

    await document_service.process_and_index_document(
        file_content=b"Sample content",
        filename="test.txt",
    )

    assert parser_thread
    assert parser_thread[0] != event_loop_thread


@pytest.mark.asyncio
async def test_process_document_embeds_chunks_in_bounded_batches(
    document_service,
    mock_vector_repo,
    mock_embedding_service,
    mock_splitter_service,
):
    mock_splitter_service.split_text.side_effect = [
        ["P" * 1000],
        [f"chunk-{idx}" for idx in range(17)],
    ]
    document_service.extract_text = lambda *_: "Sample content"
    mock_embedding_service.embed_texts.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]

    response = await document_service.process_and_index_document(
        file_content=b"Sample content",
        filename="large.pdf",
    )

    assert response.total_chunks == 17
    assert mock_embedding_service.embed_texts.call_count == 3
    assert len(mock_embedding_service.embed_texts.call_args_list[0].args[0]) == 8
    assert len(mock_embedding_service.embed_texts.call_args_list[1].args[0]) == 8
    assert len(mock_embedding_service.embed_texts.call_args_list[2].args[0]) == 1
    mock_vector_repo.save_chunks.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_document_reports_embedding_progress(document_service):
    progress = AsyncMock()
    document_service.extract_text = lambda *_: "Sample content"
    document_service.splitter_service.split_text.side_effect = [
        ["P" * 1000],
        [f"chunk-{index}" for index in range(25)],
    ]
    document_service.embedding_service.embed_texts.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]

    await document_service.process_and_index_document(
        file_content=b"Sample content",
        filename="test.pdf",
        progress_callback=progress,
    )

    batch_updates = [
        call.args[0] for call in progress.await_args_list if "batch " in call.args[0]
    ]
    assert batch_updates == [
        "Generating embeddings for test.pdf: batch 1/4.",
        "Generating embeddings for test.pdf: batch 4/4.",
    ]


def test_related_subparagraph_stays_in_parent_context():
    text = "A" * 521 + "\n\n" + "Related subparagraph."
    service = DocumentService(
        vector_repository=MagicMock(spec=VectorRepository),
        embedding_service=MagicMock(spec=EmbeddingService),
        splitter_service=LangChainRecursiveSplitter(),
    )

    chunks = service.split_into_parent_child_chunks(
        text,
        child_chunk_size=500,
        child_overlap=50,
    )

    assert len(chunks) > 1
    assert all(parent_content == text for _, parent_content, _ in chunks)
    assert all("Related subparagraph." in parent_content for _, parent_content, _ in chunks)


@pytest.mark.asyncio
async def test_process_and_index_document_rejects_missing_embeddings(
    document_service,
    mock_embedding_service,
):
    from src.core.exceptions import DocumentProcessingError

    mock_embedding_service.embed_texts.return_value = []
    with pytest.raises(DocumentProcessingError, match="does not match chunk count"):
        await document_service.process_and_index_document(
            file_content=b"Sample content",
            filename="test.txt",
        )


@pytest.mark.asyncio
async def test_process_and_index_document_rejects_empty_file(document_service):
    from src.core.exceptions import DocumentProcessingError

    with pytest.raises(DocumentProcessingError, match="empty"):
        await document_service.process_and_index_document(
            file_content=b"",
            filename="test.txt",
        )


@pytest.mark.asyncio
async def test_process_and_index_document_rejects_invalid_chunk_settings(document_service):
    from src.core.exceptions import DocumentProcessingError

    with pytest.raises(DocumentProcessingError, match="Chunk size"):
        await document_service.process_and_index_document(
            file_content=b"Sample content",
            filename="test.txt",
            chunk_size=10,
            chunk_overlap=10,
        )


@pytest.mark.asyncio
async def test_process_and_index_document_rejects_child_size_larger_than_parent(
    document_service,
):
    from src.core.exceptions import DocumentProcessingError

    with pytest.raises(DocumentProcessingError, match="Parent chunk size"):
        await document_service.process_and_index_document(
            file_content=b"Sample content",
            filename="test.txt",
            chunk_size=2500,
            chunk_overlap=50,
        )

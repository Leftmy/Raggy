import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domains.documents.schemas import DocumentChunk
from src.domains.documents.service import DocumentService
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.splitters import TextSplitterService
from src.domains.rag.service import EmbeddingService


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


def test_text_parser_extraction():
    from src.domains.documents.parsers import ParserFactory
    parser = ParserFactory.get_parser("sample.txt")
    result = parser.parse(b"Hello world from text file", "sample.txt")
    assert result == "Hello world from text file"


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
    mock_splitter_service.split_text.assert_called_once()
    mock_embedding_service.embed_texts.assert_called_once()
    mock_vector_repo.save_chunks.assert_awaited_once()

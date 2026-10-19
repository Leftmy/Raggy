import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.domains.rag.schemas import SearchResultChunk
from src.infrastructure.llm.gemini_service import GeminiLLMService
from src.core.exceptions import LLMGenerationError


@pytest.fixture
def sample_chunks():
    return [
        SearchResultChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="FastEmbed generates lightweight dense embeddings locally on CPU.",
            score=0.92,
            metadata={"filename": "architecture.txt"},
        ),
        SearchResultChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Qdrant provides scalable vector search with payload filtering.",
            score=0.88,
            metadata={"filename": "architecture.txt"},
        ),
    ]


@pytest.fixture
def mock_gemini():
    with patch("src.infrastructure.llm.gemini_service.ChatGoogleGenerativeAI") as mock_cls:
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "FastEmbed is a lightweight CPU-optimized embedding generator."
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_llm_instance
        yield mock_cls, mock_llm_instance


@pytest.mark.asyncio
async def test_generate_response_constructs_prompt_and_returns_text(mock_gemini, sample_chunks):
    mock_cls, mock_instance = mock_gemini
    service = GeminiLLMService(api_key="fake-test-key", model_name="gemini-1.5-pro")

    response = await service.generate_response(
        query="What is FastEmbed?",
        context_chunks=sample_chunks,
    )

    assert "FastEmbed is a lightweight CPU-optimized" in response
    mock_instance.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_response_empty_context_fallback(mock_gemini):
    mock_cls, mock_instance = mock_gemini
    mock_response = MagicMock()
    mock_response.content = "I could not find relevant information in the provided documents."
    mock_instance.ainvoke.return_value = mock_response

    service = GeminiLLMService(api_key="fake-test-key")
    response = await service.generate_response(query="Unknown topic", context_chunks=[])

    assert response is not None
    mock_instance.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_response_error_raises_llm_generation_error(mock_gemini, sample_chunks):
    mock_cls, mock_instance = mock_gemini
    mock_instance.ainvoke.side_effect = Exception("Google API quota exceeded")

    service = GeminiLLMService(api_key="fake-test-key")
    with pytest.raises(LLMGenerationError):
        await service.generate_response(query="Any query", context_chunks=sample_chunks)

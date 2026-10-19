import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domains.rag.pipeline import RAGPipelineService
from src.domains.rag.schemas import QueryRequest, SearchResultChunk
from src.domains.documents.schemas import DocumentChunk
from src.domains.rag.service import EmbeddingService, RerankerService, LLMService
from src.domains.documents.repositories import VectorRepository


@pytest.fixture
def mock_dependencies():
    vector_repo = MagicMock(spec=VectorRepository)
    vector_repo.search = AsyncMock()

    embedding_service = MagicMock(spec=EmbeddingService)
    embedding_service.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

    reranker_service = MagicMock(spec=RerankerService)
    reranker_service.rerank = MagicMock()

    llm_service = MagicMock(spec=LLMService)
    llm_service.generate_response = AsyncMock(return_value="Synthesized RAG Answer")

    return {
        "vector_repo": vector_repo,
        "embedding_service": embedding_service,
        "reranker_service": reranker_service,
        "llm_service": llm_service,
    }


@pytest.mark.asyncio
async def test_execute_query_full_pipeline_flow(mock_dependencies):
    vector_repo = mock_dependencies["vector_repo"]
    embedding_service = mock_dependencies["embedding_service"]
    reranker_service = mock_dependencies["reranker_service"]
    llm_service = mock_dependencies["llm_service"]

    # 1. Mock vector search results
    doc_chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            content="Qdrant search result content",
            metadata={"score": 0.85, "filename": "doc.txt"},
        )
    ]
    vector_repo.search.return_value = doc_chunks

    # 2. Mock reranker results
    reranked_chunks = [
        SearchResultChunk(
            chunk_id="c1",
            document_id="d1",
            content="Qdrant search result content",
            score=0.97,
            metadata={"filename": "doc.txt"},
        )
    ]
    reranker_service.rerank.return_value = reranked_chunks

    pipeline = RAGPipelineService(**mock_dependencies)
    request = QueryRequest(query="Tell me about Qdrant", top_k=5, rerank_top_k=2)

    response = await pipeline.execute_query(request)

    # Verifications
    embedding_service.embed_query.assert_called_once_with("Tell me about Qdrant")
    vector_repo.search.assert_awaited_once_with(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        filters=None,
    )
    reranker_service.rerank.assert_called_once()
    llm_service.generate_response.assert_awaited_once_with(
        query="Tell me about Qdrant",
        context_chunks=reranked_chunks,
    )
    assert response.query == "Tell me about Qdrant"
    assert response.answer == "Synthesized RAG Answer"
    assert len(response.source_chunks) == 1
    assert response.source_chunks[0].score == 0.97


@pytest.mark.asyncio
async def test_execute_query_when_no_documents_found(mock_dependencies):
    vector_repo = mock_dependencies["vector_repo"]
    reranker_service = mock_dependencies["reranker_service"]
    llm_service = mock_dependencies["llm_service"]

    vector_repo.search.return_value = []
    reranker_service.rerank.return_value = []

    pipeline = RAGPipelineService(**mock_dependencies)
    request = QueryRequest(query="Unknown question")

    response = await pipeline.execute_query(request)

    assert response.answer == "Synthesized RAG Answer"
    assert response.source_chunks == []
    llm_service.generate_response.assert_awaited_once_with(
        query="Unknown question",
        context_chunks=[],
    )

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domains.documents.repositories import VectorRepository
from src.domains.documents.schemas import DocumentChunk
from src.domains.rag.pipeline import RAGPipelineService
from src.domains.rag.schemas import QueryRequest, SearchResultChunk
from src.domains.rag.service import EmbeddingService, LLMService, RerankerService


def test_source_citation_includes_document_place_sender_and_upload_time():
    from src.bot.handlers.queries import format_source_citation

    chunk = SearchResultChunk(
        chunk_id="parent-1",
        document_id="doc-1",
        content="Relevant information",
        score=0.98,
        metadata={
            "filename": "fall-guide.pdf",
            "document_title": "Autumn guide",
            "place": "North Camp",
            "uploaded_by": "Owl",
            "uploaded_at": "2026-10-19T18:49:00+00:00",
        },
    )

    citation = format_source_citation(chunk)

    assert "Autumn guide" in citation
    assert "North Camp" in citation
    assert "fall-guide.pdf" in citation
    assert "Owl" in citation
    assert "2026-10-19 18:49 UTC" in citation


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
async def test_execute_query_when_no_documents_found_does_not_call_llm(mock_dependencies):
    vector_repo = mock_dependencies["vector_repo"]
    reranker_service = mock_dependencies["reranker_service"]
    llm_service = mock_dependencies["llm_service"]

    vector_repo.search.return_value = []
    reranker_service.rerank.return_value = []

    pipeline = RAGPipelineService(**mock_dependencies)
    request = QueryRequest(query="Unknown question")

    response = await pipeline.execute_query(request)

    assert "couldn't find relevant information" in response.answer
    assert response.source_chunks == []
    llm_service.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_query_expands_reranked_child_to_unique_parent_context(mock_dependencies):
    vector_repo = mock_dependencies["vector_repo"]
    reranker_service = mock_dependencies["reranker_service"]
    llm_service = mock_dependencies["llm_service"]
    parent_text = "Long parent paragraph with a relevant subparagraph."
    vector_repo.search.return_value = [
        DocumentChunk(
            chunk_id="child-1",
            document_id="doc-1",
            content="Relevant child passage",
            parent_chunk_id="parent-1",
            parent_content=parent_text,
            metadata={"filename": "doc.txt"},
            score=0.9,
        ),
        DocumentChunk(
            chunk_id="child-2",
            document_id="doc-1",
            content="Overlapping relevant child passage",
            parent_chunk_id="parent-1",
            parent_content=parent_text,
            metadata={"filename": "doc.txt"},
            score=0.8,
        ),
        DocumentChunk(
            chunk_id="child-3",
            document_id="doc-1",
            content="Another relevant passage",
            parent_chunk_id="parent-2",
            parent_content="A separate parent paragraph.",
            metadata={"filename": "doc.txt"},
            score=0.7,
        ),
    ]
    reranker_service.rerank.return_value = [
        SearchResultChunk(
            chunk_id="child-1",
            document_id="doc-1",
            content="Relevant child passage",
            score=0.98,
            metadata={"filename": "doc.txt"},
        ),
        SearchResultChunk(
            chunk_id="child-2",
            document_id="doc-1",
            content="Overlapping relevant child passage",
            score=0.95,
            metadata={"filename": "doc.txt"},
        ),
        SearchResultChunk(
            chunk_id="child-3",
            document_id="doc-1",
            content="Another relevant passage",
            score=0.9,
            metadata={"filename": "doc.txt"},
        ),
    ]

    response = await RAGPipelineService(**mock_dependencies).execute_query(
        QueryRequest(query="Find the related subparagraph", top_k=5, rerank_top_k=2)
    )

    reranker_service.rerank.assert_called_once()
    assert reranker_service.rerank.call_args.kwargs["top_n"] == 5
    llm_service.generate_response.assert_awaited_once()
    context_chunks = llm_service.generate_response.await_args.kwargs["context_chunks"]
    assert len(context_chunks) == 2
    assert context_chunks[0].chunk_id == "parent-1"
    assert context_chunks[0].content == parent_text
    assert context_chunks[0].score == 0.98
    assert context_chunks[1].chunk_id == "parent-2"
    assert context_chunks[1].content == "A separate parent paragraph."
    assert response.source_chunks == context_chunks

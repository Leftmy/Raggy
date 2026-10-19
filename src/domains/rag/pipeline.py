import asyncio
from typing import List

from src.domains.documents.repositories import VectorRepository
from src.domains.rag.schemas import QueryRequest, QueryResponse, SearchResultChunk
from src.domains.rag.service import EmbeddingService, LLMService, RerankerService


class RAGPipelineService:
    def __init__(
        self,
        vector_repo: VectorRepository,
        embedding_service: EmbeddingService,
        reranker_service: RerankerService,
        llm_service: LLMService,
    ):
        self.vector_repo = vector_repo
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.llm_service = llm_service

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """
        1. Embed the user query (EmbeddingService)
        2. Vector search candidate chunks in Vector Store (top_k)
        3. Rerank retrieved candidate chunks (RerankerService)
        4. Synthesize context and prompt LLM (LLMService)
        5. Return structured answer and source chunks
        """
        # 1. Embed query
        query_vector = await asyncio.to_thread(self.embedding_service.embed_query, request.query)

        # 2. Vector search candidates
        retrieved_chunks = await self.vector_repo.search(
            query_vector=query_vector,
            top_k=request.top_k,
            filters=request.filter_metadata,
        )

        # Map DocumentChunk to SearchResultChunk for reranker
        chunks_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        candidate_search_chunks: List[SearchResultChunk] = [
            SearchResultChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=chunk.score if chunk.score is not None else 0.0,
                metadata=chunk.metadata,
            )
            for chunk in retrieved_chunks
        ]

        # 3. Rerank candidate chunks
        reranked_chunks: List[SearchResultChunk] = []
        if candidate_search_chunks:
            reranked_chunks = await asyncio.to_thread(
                self.reranker_service.rerank,
                query=request.query,
                documents=candidate_search_chunks,
                top_n=request.top_k,
            )

        # Expand matched child chunks to their parent context and avoid duplicate parents.
        expanded_chunks: List[SearchResultChunk] = []
        seen_parent_ids = set()
        for chunk in reranked_chunks:
            retrieved = chunks_by_id.get(chunk.chunk_id)
            parent_id = retrieved.parent_chunk_id if retrieved else None
            expanded_chunk_id = parent_id or chunk.chunk_id
            if expanded_chunk_id in seen_parent_ids:
                continue
            seen_parent_ids.add(expanded_chunk_id)
            expanded_chunks.append(
                SearchResultChunk(
                    chunk_id=expanded_chunk_id,
                    document_id=chunk.document_id,
                    content=(
                        retrieved.parent_content
                        if retrieved and retrieved.parent_content
                        else chunk.content
                    ),
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
            )
            if len(expanded_chunks) >= request.rerank_top_k:
                break

        # 4. Generate response via LLM
        if expanded_chunks:
            answer = await self.llm_service.generate_response(
                query=request.query,
                context_chunks=expanded_chunks,
            )
        else:
            answer = "I couldn't find relevant information in the indexed documents."

        # 5. Return complete QueryResponse
        return QueryResponse(
            query=request.query,
            answer=answer,
            source_chunks=expanded_chunks,
        )

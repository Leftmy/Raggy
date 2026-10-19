from typing import List
from src.domains.rag.schemas import QueryRequest, QueryResponse, SearchResultChunk
from src.domains.rag.service import EmbeddingService, RerankerService, LLMService
from src.domains.documents.repositories import VectorRepository


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
        query_vector = self.embedding_service.embed_query(request.query)

        # 2. Vector search candidates
        retrieved_chunks = await self.vector_repo.search(
            query_vector=query_vector,
            top_k=request.top_k,
            filters=request.filter_metadata,
        )

        # Map DocumentChunk to SearchResultChunk for reranker
        candidate_search_chunks: List[SearchResultChunk] = [
            SearchResultChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=float(chunk.metadata.get("score", 0.0)),
                metadata=chunk.metadata,
            )
            for chunk in retrieved_chunks
        ]

        # 3. Rerank candidate chunks
        reranked_chunks: List[SearchResultChunk] = []
        if candidate_search_chunks:
            reranked_chunks = self.reranker_service.rerank(
                query=request.query,
                documents=candidate_search_chunks,
                top_n=request.rerank_top_k,
            )

        # 4. Generate response via LLM
        answer = await self.llm_service.generate_response(
            query=request.query,
            context_chunks=reranked_chunks,
        )

        # 5. Return complete QueryResponse
        return QueryResponse(
            query=request.query,
            answer=answer,
            source_chunks=reranked_chunks,
        )

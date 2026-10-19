import asyncio
import re
import unicodedata
from typing import List

from src.domains.documents.repositories import VectorRepository
from src.domains.documents.schemas import DocumentChunk
from src.domains.rag.schemas import QueryRequest, QueryResponse, SearchResultChunk
from src.domains.rag.service import EmbeddingService, LLMService, RerankerService

_FILENAME_PATTERN = re.compile(r"(?<![\w.])[\w.-]+\.(?:pdf|docx|txt|md)\b", re.IGNORECASE)


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
        seen_chunk_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        for filename in dict.fromkeys(_FILENAME_PATTERN.findall(request.query)):
            filters = dict(request.filter_metadata or {})
            if "filename" in filters and filters["filename"] != filename:
                continue
            filters["filename"] = filename
            filename_chunks = await self.vector_repo.search(
                query_vector=query_vector,
                top_k=request.top_k,
                filters=filters,
            )
            retrieved_chunks.extend(
                chunk for chunk in filename_chunks if chunk.chunk_id not in seen_chunk_ids
            )
            seen_chunk_ids.update(chunk.chunk_id for chunk in filename_chunks)

        explicit_targets = self._explicit_target_chunks(request.query, retrieved_chunks)
        if explicit_targets:
            retrieved_chunks = explicit_targets

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

    @staticmethod
    def _explicit_target_chunks(
        query: str,
        chunks: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        normalized_query = RAGPipelineService._normalize_target(query)
        matching_chunks = []
        for chunk in chunks:
            target_values = (
                chunk.metadata.get("filename"),
                chunk.metadata.get("document_title"),
            )
            if any(
                normalized_target
                and f" {normalized_target} " in f" {normalized_query} "
                for target in target_values
                if target
                for normalized_target in [
                    RAGPipelineService._normalize_target(str(target))
                ]
            ):
                matching_chunks.append(chunk)
        return matching_chunks

    @staticmethod
    def _normalize_target(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(re.sub(r"[^\w]+", " ", normalized).split())

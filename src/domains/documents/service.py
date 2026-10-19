import asyncio
import logging
import uuid
from typing import List, Optional

from src.core.exceptions import DocumentProcessingError
from src.domains.documents.parsers import ParserFactory, sanitize_filename
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.schemas import DocumentChunk, DocumentUploadResponse
from src.domains.documents.splitters import TextSplitterService
from src.domains.rag.service import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        splitter_service: TextSplitterService,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_chunks: int = 10000,
        max_upload_size: int = 20 * 1024 * 1024,
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.splitter_service = splitter_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.max_upload_size = max_upload_size

    def extract_text(self, file_content: bytes, filename: str) -> str:
        parser = ParserFactory.get_parser(filename)
        return parser.parse(file_content, filename)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        return self.splitter_service.split_text(text, chunk_size=chunk_size, overlap=overlap)

    async def process_and_index_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> DocumentUploadResponse:
        """Parse, chunk, generate embeddings, and persist in vector repository."""
        filename = sanitize_filename(filename)
        if not filename:
            raise DocumentProcessingError("A valid filename with a supported extension is required")
        if not file_content:
            raise DocumentProcessingError("The uploaded document is empty")
        if len(file_content) > self.max_upload_size:
            raise DocumentProcessingError("The uploaded document exceeds the size limit")

        selected_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        selected_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        if selected_chunk_size <= 0 or selected_overlap < 0 or selected_overlap >= selected_chunk_size:
            raise DocumentProcessingError("Chunk size must be positive and overlap must be smaller than chunk size")
        doc_id = document_id or str(uuid.uuid4())
        raw_text = self.extract_text(file_content, filename)
        if not raw_text.strip():
            raise DocumentProcessingError("The document contains no extractable text")

        step = selected_chunk_size - selected_overlap
        estimated_chunks = (len(raw_text) + step - 1) // step
        if estimated_chunks > self.max_chunks:
            raise DocumentProcessingError(
                f"The document is estimated to exceed the maximum of {self.max_chunks} chunks"
            )
        chunks_text = await asyncio.to_thread(
            self.chunk_text,
            raw_text,
            chunk_size=selected_chunk_size,
            overlap=selected_overlap,
        )
        if not chunks_text:
            raise DocumentProcessingError("The document produced no text chunks")
        if len(chunks_text) > self.max_chunks:
            raise DocumentProcessingError(
                f"The document exceeds the maximum of {self.max_chunks} chunks"
            )

        embeddings = await asyncio.to_thread(self.embedding_service.embed_texts, chunks_text)
        if len(embeddings) != len(chunks_text):
            raise DocumentProcessingError(
                f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks_text)})"
            )
        if any(not embedding for embedding in embeddings):
            raise DocumentProcessingError("Embedding service returned an empty vector")

        document_chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}_{idx}",
                document_id=doc_id,
                content=chunk_content,
                metadata={"filename": filename, "chunk_index": idx},
                embedding=embeddings[idx],
            )
            for idx, chunk_content in enumerate(chunks_text)
        ]

        await self.vector_repository.save_chunks(document_chunks)
        logger.info("Indexed document %s with %d chunks", doc_id, len(document_chunks))

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            total_chunks=len(document_chunks),
        )

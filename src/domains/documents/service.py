import asyncio
import logging
import unicodedata
import uuid
from datetime import datetime, timezone
from math import ceil
from time import perf_counter
from typing import Awaitable, Callable, List, Optional

from src.core.exceptions import DocumentProcessingError
from src.domains.documents.parsers import ParserFactory, sanitize_filename
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.schemas import DocumentChunk, DocumentUploadResponse
from src.domains.documents.splitters import TextSplitterService
from src.domains.rag.service import EmbeddingService

logger = logging.getLogger(__name__)
EMBEDDING_BATCH_SIZE = 8


class DocumentService:
    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        splitter_service: TextSplitterService,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
        max_chunks: int = 10000,
        max_upload_size: int = 20 * 1024 * 1024,
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.splitter_service = splitter_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.max_chunks = max_chunks
        self.max_upload_size = max_upload_size

    def extract_text(self, file_content: bytes, filename: str) -> str:
        parser = ParserFactory.get_parser(filename)
        return parser.parse(file_content, filename)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        return self.splitter_service.split_text(text, chunk_size=chunk_size, overlap=overlap)

    def split_into_parent_child_chunks(
        self,
        text: str,
        child_chunk_size: int,
        child_overlap: int,
    ) -> List[tuple[int, str, str]]:
        parent_chunks = self.splitter_service.split_text(
            text,
            chunk_size=self.parent_chunk_size,
            overlap=0,
        )
        chunks = []
        for parent_index, parent_content in enumerate(parent_chunks):
            child_contents = self.chunk_text(
                parent_content,
                chunk_size=child_chunk_size,
                overlap=child_overlap,
            )
            chunks.extend(
                (parent_index, parent_content, child_content)
                for child_content in child_contents
            )
        return chunks

    async def process_and_index_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        document_title: Optional[str] = None,
        place: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        uploaded_by_username: Optional[str] = None,
        uploaded_by_id: Optional[int] = None,
        uploaded_at: Optional[datetime] = None,
        progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> DocumentUploadResponse:
        """Parse, chunk, generate embeddings, and persist in vector repository."""
        filename = sanitize_filename(filename)
        if not filename:
            raise DocumentProcessingError("A valid filename with a supported extension is required")
        if not file_content:
            raise DocumentProcessingError("The uploaded document is empty")
        if len(file_content) > self.max_upload_size:
            raise DocumentProcessingError("The uploaded document exceeds the size limit")
        if (document_title is None) != (place is None):
            raise DocumentProcessingError("Both document title and place are required")
        if document_title is not None:
            document_title = document_title.strip()
            place = (place or "").strip()
            if not document_title or not place:
                raise DocumentProcessingError("Both document title and place are required")
            if uploaded_at is None:
                raise DocumentProcessingError("Upload time is required for document replacement")
            if uploaded_at.tzinfo is None:
                uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
            else:
                uploaded_at = uploaded_at.astimezone(timezone.utc)

        selected_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        selected_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        if selected_chunk_size <= 0 or selected_overlap < 0 or selected_overlap >= selected_chunk_size:
            raise DocumentProcessingError("Chunk size must be positive and overlap must be smaller than chunk size")
        if self.parent_chunk_size < selected_chunk_size:
            raise DocumentProcessingError("Parent chunk size must be at least the child chunk size")
        doc_id = document_id or str(uuid.uuid4())
        stage_started = perf_counter()
        raw_text = await asyncio.to_thread(self.extract_text, file_content, filename)
        logger.info(
            "Parsed document %s: %d characters in %.2f seconds",
            filename,
            len(raw_text),
            perf_counter() - stage_started,
        )
        if progress_callback:
            await progress_callback(f"Read {len(raw_text):,} characters from {filename}. Splitting text...")
        if not raw_text.strip():
            raise DocumentProcessingError("The document contains no extractable text")

        stage_started = perf_counter()
        parent_child_chunks = await asyncio.to_thread(
            self.split_into_parent_child_chunks,
            raw_text,
            child_chunk_size=selected_chunk_size,
            child_overlap=selected_overlap,
        )
        if not parent_child_chunks:
            raise DocumentProcessingError("The document produced no text chunks")
        if len(parent_child_chunks) > self.max_chunks:
            raise DocumentProcessingError(
                f"The document exceeds the maximum of {self.max_chunks} chunks"
            )
        logger.info(
            "Split document %s into %d chunks in %.2f seconds",
            filename,
            len(parent_child_chunks),
            perf_counter() - stage_started,
        )
        if progress_callback:
            await progress_callback(
                f"Split {filename} into {len(parent_child_chunks):,} chunks. Generating embeddings..."
            )

        chunks_text = [child_content for _, _, child_content in parent_child_chunks]
        embeddings = []
        batch_count = ceil(len(chunks_text) / EMBEDDING_BATCH_SIZE)
        stage_started = perf_counter()
        for batch_index, offset in enumerate(
            range(0, len(chunks_text), EMBEDDING_BATCH_SIZE),
            start=1,
        ):
            batch_embeddings = await asyncio.to_thread(
                self.embedding_service.embed_texts,
                chunks_text[offset:offset + EMBEDDING_BATCH_SIZE],
            )
            embeddings.extend(batch_embeddings)
            if progress_callback and (
                batch_index == 1 or batch_index % 10 == 0 or batch_index == batch_count
            ):
                await progress_callback(
                    f"Generating embeddings for {filename}: batch {batch_index}/{batch_count}."
                )
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == batch_count:
                logger.info(
                    "Embedded batch %d/%d for %s",
                    batch_index,
                    batch_count,
                    filename,
                )
        logger.info(
            "Generated embeddings for %s in %.2f seconds",
            filename,
            perf_counter() - stage_started,
        )
        if len(embeddings) != len(chunks_text):
            raise DocumentProcessingError(
                f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks_text)})"
            )
        if any(not embedding for embedding in embeddings):
            raise DocumentProcessingError("Embedding service returned an empty vector")

        document_chunks = []
        for idx, (parent_index, parent_content, chunk_content) in enumerate(parent_child_chunks):
            metadata = {
                "filename": filename,
                "chunk_index": idx,
                "parent_chunk_index": parent_index,
            }
            if document_title is not None and place is not None and uploaded_at is not None:
                metadata.update(
                    {
                        "document_title": document_title,
                        "place": place,
                        "document_identity_key": self._normalize_identity(document_title),
                        "place_key": self._normalize_identity(place),
                        "uploaded_at": uploaded_at.isoformat(),
                    }
                )
                if uploaded_by:
                    metadata["uploaded_by"] = uploaded_by
                if uploaded_by_username:
                    metadata["uploaded_by_username"] = uploaded_by_username
                if uploaded_by_id is not None:
                    metadata["uploaded_by_id"] = uploaded_by_id
            document_chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_{idx}",
                    document_id=doc_id,
                    content=chunk_content,
                    parent_chunk_id=f"{doc_id}_parent_{parent_index}",
                    parent_content=parent_content,
                    metadata=metadata,
                    embedding=embeddings[idx],
                )
            )

        replaced_previous = False
        stage_started = perf_counter()
        if document_title is not None and place is not None and uploaded_at is not None:
            accepted, replaced_previous = await self.vector_repository.replace_document(
                document_chunks,
                document_identity_key=self._normalize_identity(document_title),
                place_key=self._normalize_identity(place),
                uploaded_at=uploaded_at,
            )
            if not accepted:
                raise DocumentProcessingError(
                    "A newer version of this document for this place is already indexed"
                )
        else:
            await self.vector_repository.save_chunks(document_chunks)
        logger.info(
            "Indexed document %s (%s) with %d chunks in %.2f seconds",
            doc_id,
            filename,
            len(document_chunks),
            perf_counter() - stage_started,
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            total_chunks=len(document_chunks),
            replaced_previous=replaced_previous,
        )

    @staticmethod
    def _normalize_identity(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        return " ".join(normalized.split()).casefold()

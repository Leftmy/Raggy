from typing import List, Optional
import uuid
from src.domains.documents.schemas import DocumentChunk, DocumentUploadResponse
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.parsers import ParserFactory
from src.domains.documents.splitters import TextSplitterService
from src.domains.rag.service import EmbeddingService


class DocumentService:
    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        splitter_service: TextSplitterService,
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.splitter_service = splitter_service

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
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> DocumentUploadResponse:
        """Parse, chunk, generate embeddings, and persist in vector repository."""
        doc_id = document_id or str(uuid.uuid4())
        raw_text = self.extract_text(file_content, filename)
        chunks_text = self.chunk_text(raw_text, chunk_size=chunk_size, overlap=chunk_overlap)
        
        embeddings = self.embedding_service.embed_texts(chunks_text)
        
        document_chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}_{idx}",
                document_id=doc_id,
                content=chunk_content,
                metadata={"filename": filename, "chunk_index": idx},
                embedding=embeddings[idx] if idx < len(embeddings) else None,
            )
            for idx, chunk_content in enumerate(chunks_text)
        ]

        await self.vector_repository.save_chunks(document_chunks)

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            total_chunks=len(document_chunks),
        )

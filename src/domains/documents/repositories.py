from abc import ABC, abstractmethod
from typing import List, Optional
from src.domains.documents.schemas import DocumentChunk


class VectorRepository(ABC):
    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the repository."""
        pass

    @abstractmethod
    async def save_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Persist chunk embeddings in vector DB."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[DocumentChunk]:
        """Perform vector similarity search."""
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> bool:
        """Remove document vectors from storage."""
        pass

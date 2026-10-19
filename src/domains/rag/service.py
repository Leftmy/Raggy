from abc import ABC, abstractmethod
from typing import List
from src.domains.rag.schemas import SearchResultChunk


class EmbeddingService(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass


class RerankerService(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[SearchResultChunk], top_n: int) -> List[SearchResultChunk]:
        pass


class LLMService(ABC):
    @abstractmethod
    async def generate_response(self, query: str, context_chunks: List[SearchResultChunk]) -> str:
        pass

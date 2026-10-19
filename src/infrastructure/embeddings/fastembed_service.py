from typing import List

from fastembed import TextEmbedding

from src.core.exceptions import EmbeddingError
from src.domains.rag.service import EmbeddingService


class FastEmbedService(EmbeddingService):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        try:
            self._model: TextEmbedding = TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            raise EmbeddingError("Failed to initialize the embedding model") from exc

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return [list(vec) for vec in self._model.embed(texts)]
        except Exception as exc:
            raise EmbeddingError("Failed to generate text embeddings") from exc

    def embed_query(self, query: str) -> List[float]:
        if not query.strip():
            raise ValueError("query must not be empty")
        vectors = self.embed_texts([query])
        if len(vectors) != 1 or not vectors[0]:
            raise EmbeddingError("Embedding model returned no vector for the query")
        return vectors[0]

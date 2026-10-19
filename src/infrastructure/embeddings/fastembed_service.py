from typing import List, Optional
from fastembed import TextEmbedding
from src.domains.rag.service import EmbeddingService


class FastEmbedService(EmbeddingService):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model: Optional[TextEmbedding] = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [list(vec) for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else []

import logging
from typing import List

from flashrank import Ranker, RerankRequest

from src.core.exceptions import RerankerError
from src.domains.rag.schemas import SearchResultChunk
from src.domains.rag.service import RerankerService

logger = logging.getLogger(__name__)


class FlashRankReranker(RerankerService):
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        try:
            self._ranker = Ranker(model_name=self.model_name)
        except Exception as exc:
            raise RerankerError("Failed to initialize the reranker model") from exc

    def rerank(
        self,
        query: str,
        documents: List[SearchResultChunk],
        top_n: int = 3,
    ) -> List[SearchResultChunk]:
        if not documents:
            return []
        if top_n < 1:
            raise ValueError("top_n must be greater than zero")
        if len({doc.chunk_id for doc in documents}) != len(documents):
            raise RerankerError("Candidate chunk IDs must be unique")

        # Build map for fast chunk lookup by id
        chunk_map = {doc.chunk_id: doc for doc in documents}
        passages = [{"id": doc.chunk_id, "text": doc.content} for doc in documents]

        rerank_request = RerankRequest(query=query, passages=passages)
        try:
            results = self._ranker.rerank(rerank_request)
        except Exception as exc:
            logger.error("FlashRank reranking failed (%s)", type(exc).__name__)
            raise RerankerError("Failed to rerank candidate documents") from exc

        reranked_chunks: List[SearchResultChunk] = []
        for res in results:
            cid = res.get("id")
            original_doc = chunk_map.get(cid)
            if original_doc:
                reranked_chunks.append(
                    SearchResultChunk(
                        chunk_id=original_doc.chunk_id,
                        document_id=original_doc.document_id,
                        content=original_doc.content,
                        score=float(res.get("score", 0.0)),
                        metadata=original_doc.metadata,
                    )
                )
                if len(reranked_chunks) >= top_n:
                    break

        return reranked_chunks

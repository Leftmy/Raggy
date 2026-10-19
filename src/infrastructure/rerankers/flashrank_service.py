from typing import List, Optional
from flashrank import Ranker, RerankRequest
from src.domains.rag.service import RerankerService
from src.domains.rag.schemas import SearchResultChunk


class FlashRankReranker(RerankerService):
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker: Optional[Ranker] = None

    def _get_ranker(self) -> Ranker:
        if self._ranker is None:
            self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def rerank(
        self,
        query: str,
        documents: List[SearchResultChunk],
        top_n: int = 3,
    ) -> List[SearchResultChunk]:
        if not documents:
            return []

        ranker = self._get_ranker()

        # Build map for fast chunk lookup by id
        chunk_map = {doc.chunk_id: doc for doc in documents}
        passages = [{"id": doc.chunk_id, "text": doc.content} for doc in documents]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        reranked_chunks: List[SearchResultChunk] = []
        for res in results[:top_n]:
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

        return reranked_chunks

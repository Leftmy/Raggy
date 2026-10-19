from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import RerankerError
from src.domains.rag.schemas import SearchResultChunk
from src.infrastructure.rerankers.flashrank_service import FlashRankReranker


@pytest.fixture
def sample_candidates():
    return [
        SearchResultChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Document about dogs and pets.",
            score=0.7,
            metadata={"source": "pets.txt"},
        ),
        SearchResultChunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            content="Document about machine learning and artificial intelligence.",
            score=0.6,
            metadata={"source": "ml.txt"},
        ),
    ]


@pytest.fixture
def mock_flashrank():
    with patch("src.infrastructure.rerankers.flashrank_service.Ranker") as mock_ranker_cls:
        ranker_instance = MagicMock()
        # FlashRank rerank returns list of dicts with 'id', 'text', 'score', etc.
        ranker_instance.rerank.return_value = [
            {"id": "chunk_2", "text": "Document about machine learning and artificial intelligence.", "score": 0.98},
            {"id": "chunk_1", "text": "Document about dogs and pets.", "score": 0.12},
        ]
        mock_ranker_cls.return_value = ranker_instance
        yield mock_ranker_cls, ranker_instance


def test_rerank_sorts_by_relevance_score(mock_flashrank, sample_candidates):
    mock_cls, mock_instance = mock_flashrank
    reranker = FlashRankReranker(model_name="ms-marco-TinyBERT-L-2-v2")

    mock_cls.assert_called_once_with(model_name="ms-marco-TinyBERT-L-2-v2")
    reranked = reranker.rerank(query="What is artificial intelligence?", documents=sample_candidates, top_n=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "chunk_2"
    assert reranked[0].score == 0.98
    assert reranked[1].chunk_id == "chunk_1"
    assert reranked[1].score == 0.12


def test_rerank_limits_to_top_n(mock_flashrank, sample_candidates):
    reranker = FlashRankReranker()
    reranked = reranker.rerank(query="AI query", documents=sample_candidates, top_n=1)

    assert len(reranked) == 1
    assert reranked[0].chunk_id == "chunk_2"


def test_rerank_empty_candidates(mock_flashrank):
    reranker = FlashRankReranker()
    reranked = reranker.rerank(query="test", documents=[], top_n=3)
    assert reranked == []


def test_initialization_failure_raises_reranker_error():
    with patch(
        "src.infrastructure.rerankers.flashrank_service.Ranker",
        side_effect=RuntimeError("model unavailable"),
    ):
        with pytest.raises(RerankerError, match="Failed to initialize"):
            FlashRankReranker()

import pytest
from pydantic import ValidationError
from src.domains.rag.schemas import QueryRequest


def test_query_request_strips_whitespace_and_accepts_valid_limits():
    request = QueryRequest(query="  What is RAG?  ", top_k=10, rerank_top_k=3)
    assert request.query == "What is RAG?"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "question", "top_k": 0},
        {"query": "question", "top_k": 2, "rerank_top_k": 3},
        {"query": "question", "filter_metadata": {"nested": {"value": 1}}},
    ],
)
def test_query_request_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        QueryRequest(**payload)

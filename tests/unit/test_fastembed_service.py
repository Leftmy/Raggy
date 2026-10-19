import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.embeddings.fastembed_service import FastEmbedService


@pytest.fixture
def mock_text_embedding():
    with patch("src.infrastructure.embeddings.fastembed_service.TextEmbedding") as mock_cls:
        mock_instance = MagicMock()
        # FastEmbed embed() returns an iterable/generator of numpy arrays
        mock_instance.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


def test_embed_texts_returns_float_vectors(mock_text_embedding):
    mock_cls, mock_instance = mock_text_embedding
    service = FastEmbedService(model_name="BAAI/bge-small-en-v1.5")
    
    texts = ["First chunk", "Second chunk"]
    vectors = service.embed_texts(texts)

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]
    mock_instance.embed.assert_called_once_with(texts)


def test_embed_texts_empty_returns_empty_list():
    service = FastEmbedService()
    vectors = service.embed_texts([])
    assert vectors == []


def test_embed_query_returns_single_vector(mock_text_embedding):
    mock_cls, mock_instance = mock_text_embedding
    mock_instance.embed.return_value = [[0.1, 0.2, 0.3]]

    service = FastEmbedService()
    query_vector = service.embed_query("What is RAG?")

    assert query_vector == [0.1, 0.2, 0.3]
    mock_instance.embed.assert_called_once_with(["What is RAG?"])

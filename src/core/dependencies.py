from functools import lru_cache
from src.core.config import get_settings
from src.domains.documents.repositories import VectorRepository
from src.domains.documents.splitters import TextSplitterService
from src.domains.documents.service import DocumentService
from src.domains.rag.service import EmbeddingService, RerankerService, LLMService
from src.domains.rag.pipeline import RAGPipelineService

from src.infrastructure.qdrant.repository import QdrantVectorRepository
from src.infrastructure.embeddings.fastembed_service import FastEmbedService
from src.infrastructure.rerankers.flashrank_service import FlashRankReranker
from src.infrastructure.llm.gemini_service import GeminiLLMService
from src.infrastructure.splitters.langchain_splitter import LangChainRecursiveSplitter


@lru_cache
def get_vector_repository() -> VectorRepository:
    settings = get_settings()
    # Factory logic based on VECTOR_STORE_PROVIDER
    if settings.VECTOR_STORE_PROVIDER.lower() == "qdrant":
        return QdrantVectorRepository(
            host=settings.VECTOR_STORE_HOST,
            port=settings.VECTOR_STORE_PORT,
            collection_name=settings.VECTOR_STORE_COLLECTION,
            api_key=settings.VECTOR_STORE_API_KEY,
        )
    raise ValueError(f"Unsupported vector store provider: {settings.VECTOR_STORE_PROVIDER}")


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    # Factory logic based on EMBEDDING_PROVIDER
    if settings.EMBEDDING_PROVIDER.lower() == "fastembed":
        return FastEmbedService(model_name=settings.EMBEDDING_MODEL)
    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")


@lru_cache
def get_reranker_service() -> RerankerService:
    settings = get_settings()
    # Factory logic based on RERANKER_PROVIDER
    if settings.RERANKER_PROVIDER.lower() == "flashrank":
        return FlashRankReranker(model_name=settings.RERANKER_MODEL)
    raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")


@lru_cache
def get_llm_service() -> LLMService:
    settings = get_settings()
    # Factory logic based on LLM_PROVIDER
    if settings.LLM_PROVIDER.lower() == "gemini":
        return GeminiLLMService(
            api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


@lru_cache
def get_splitter_service() -> TextSplitterService:
    return LangChainRecursiveSplitter()


def get_document_service() -> DocumentService:
    return DocumentService(
        vector_repository=get_vector_repository(),
        embedding_service=get_embedding_service(),
        splitter_service=get_splitter_service(),
    )


def get_rag_pipeline_service() -> RAGPipelineService:
    return RAGPipelineService(
        vector_repo=get_vector_repository(),
        embedding_service=get_embedding_service(),
        reranker_service=get_reranker_service(),
        llm_service=get_llm_service(),
    )

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = ""

    # LLM Settings
    LLM_PROVIDER: str = "gemini"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-pro"

    # Vector Store Settings
    VECTOR_STORE_PROVIDER: str = "qdrant"
    VECTOR_STORE_HOST: str = "localhost"
    VECTOR_STORE_PORT: int = 6333
    VECTOR_STORE_GRPC_PORT: int = 6334
    VECTOR_STORE_COLLECTION: str = "rag_documents"
    VECTOR_STORE_API_KEY: str | None = None

    # Embeddings & Reranking Settings
    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    RERANKER_PROVIDER: str = "flashrank"
    RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"

    # Chunking & Retrieval Strategy
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

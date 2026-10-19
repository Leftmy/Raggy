from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    API_AUTH_TOKEN: SecretStr | None = None
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=20 * 1024 * 1024, gt=0)
    API_RATE_LIMIT_PER_HOUR: int = Field(default=60, gt=0)
    BOT_RATE_LIMIT_PER_HOUR: int = Field(default=60, gt=0)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60 * 60, gt=0)

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: SecretStr | None = None
    TELEGRAM_ALLOWED_USER_IDS: list[int] = Field(default_factory=list)

    # LLM Settings
    LLM_PROVIDER: str = "gemini"
    LLM_API_KEY: SecretStr | None = None
    LLM_MODEL: str = "gemini-3.8-flash"
    LLM_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    LLM_SYSTEM_PROMPT: str | None = None

    # Vector Store Settings
    VECTOR_STORE_PROVIDER: str = "qdrant"
    VECTOR_STORE_HOST: str = "localhost"
    VECTOR_STORE_PORT: int = 6333
    VECTOR_STORE_GRPC_PORT: int = 6334
    VECTOR_STORE_COLLECTION: str = "rag_documents"
    VECTOR_STORE_API_KEY: SecretStr | None = None

    # Embeddings & Reranking Settings
    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    RERANKER_PROVIDER: str = "flashrank"
    RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"

    # Chunking & Retrieval Strategy
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=50, ge=0)
    PARENT_CHUNK_SIZE: int = Field(default=2000, gt=0)
    MAX_CHUNKS_PER_DOCUMENT: int = Field(default=10000, gt=0)
    TOP_K_RETRIEVAL: int = Field(default=10, ge=1, le=100)
    TOP_K_RERANK: int = Field(default=3, ge=1, le=100)

    @field_validator("API_AUTH_TOKEN", "TELEGRAM_BOT_TOKEN", "LLM_API_KEY", "VECTOR_STORE_API_KEY", mode="before")
    @classmethod
    def normalize_empty_secrets(cls, value):
        return None if value == "" else value

    @field_validator("LOG_LEVEL")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be a valid Python logging level")
        return level

    @model_validator(mode="after")
    def validate_runtime_settings(self):
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.PARENT_CHUNK_SIZE < self.CHUNK_SIZE:
            raise ValueError("PARENT_CHUNK_SIZE must be at least CHUNK_SIZE")
        if self.TOP_K_RERANK > self.TOP_K_RETRIEVAL:
            raise ValueError("TOP_K_RERANK cannot exceed TOP_K_RETRIEVAL")
        if self.APP_ENV == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be false in production")
            if self.API_AUTH_TOKEN is None:
                raise ValueError("API_AUTH_TOKEN is required in production")
            if self.LLM_PROVIDER.lower() == "gemini" and not self.LLM_API_KEY:
                raise ValueError("LLM_API_KEY is required for Gemini in production")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from src.core.config import get_settings


class SearchResultChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default_factory=lambda: get_settings().TOP_K_RETRIEVAL, ge=1, le=100)
    rerank_top_k: int = Field(default_factory=lambda: get_settings().TOP_K_RERANK, ge=1, le=100)
    filter_metadata: Optional[Dict[str, Any]] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value

    @model_validator(mode="after")
    def validate_rerank_count(self):
        if self.rerank_top_k > self.top_k:
            raise ValueError("rerank_top_k cannot exceed top_k")
        for value in (self.filter_metadata or {}).values():
            if isinstance(value, (dict, list, set)):
                raise ValueError("filter_metadata values must be scalar")
        return self


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_chunks: List[SearchResultChunk]

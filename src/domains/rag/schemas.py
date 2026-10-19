from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SearchResultChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank_top_k: int = 3
    filter_metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_chunks: List[SearchResultChunk]

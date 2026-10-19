from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    score: Optional[float] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

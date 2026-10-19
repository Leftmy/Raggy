from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    parent_chunk_id: Optional[str] = None
    parent_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    score: Optional[float] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

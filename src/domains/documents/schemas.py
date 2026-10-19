from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


class DocumentUploadRequest(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

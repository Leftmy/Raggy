from fastapi import APIRouter, UploadFile, File, Depends, status
from src.domains.documents.schemas import DocumentUploadResponse
from src.domains.documents.service import DocumentService
from src.core.dependencies import get_document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
):
    """Upload document (.pdf, .docx, .txt), chunk, and index into vector store."""
    content = await file.read()
    response = await document_service.process_and_index_document(
        file_content=content,
        filename=file.filename or "unknown.txt",
    )
    return response

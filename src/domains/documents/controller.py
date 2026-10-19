from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from src.domains.documents.schemas import DocumentUploadResponse
from src.domains.documents.service import DocumentService
from src.core.dependencies import get_document_service
from src.core.config import get_settings
from src.core.exceptions import DocumentProcessingError
from src.core.security import require_api_auth
from src.domains.documents.parsers import sanitize_filename

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(require_api_auth)],
)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
):
    """Upload a supported document, chunk it, and index it into the vector store."""
    filename = sanitize_filename(file.filename or "")
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not filename or extension not in {"pdf", "docx", "txt", "md"}:
        await file.close()
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")

    try:
        content = await file.read(get_settings().MAX_UPLOAD_SIZE_BYTES + 1)
        if len(content) > get_settings().MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds the upload limit")
        try:
            return await document_service.process_and_index_document(
                file_content=content,
                filename=filename,
            )
        except DocumentProcessingError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message) from exc
    finally:
        await file.close()

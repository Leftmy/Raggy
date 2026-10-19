from aiogram import Router, types, F
from aiogram.enums import ContentType
from src.domains.documents.service import DocumentService

router = Router(name="document_handlers")


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_document_upload(
    message: types.Message,
    document_service: DocumentService,
):
    """
    Handle user document uploads (.pdf, .docx, .txt).
    Downloads file, sends to DocumentService -> VectorRepository (FastEmbed -> Qdrant).
    """
    document = message.document
    if not document:
        return

    file_name = document.file_name or "unknown.txt"
    status_msg = await message.reply(f"Downloading and indexing: {file_name}...")

    try:
        # Download document buffer
        file_io = await message.bot.download(document.file_id)
        if file_io:
            file_bytes = file_io.read()
            response = await document_service.process_and_index_document(
                file_content=file_bytes,
                filename=file_name,
            )
            await status_msg.edit_text(
                f"Document successfully indexed into the knowledge base.\n"
                f"File: {response.filename}\n"
                f"Total chunks: {response.total_chunks}"
            )
        else:
            await status_msg.edit_text("Error: Failed to retrieve file from Telegram.")
    except Exception as e:
        await status_msg.edit_text(f"Error processing document: {str(e)}")

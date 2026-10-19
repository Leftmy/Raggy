import logging

from aiogram import F, Router, types
from aiogram.enums import ContentType

from src.bot.handlers.base import is_authorized
from src.core.config import Settings
from src.core.rate_limit import InMemoryRateLimiter
from src.domains.documents.parsers import sanitize_filename
from src.domains.documents.service import DocumentService

router = Router(name="document_handlers")
logger = logging.getLogger(__name__)


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_document_upload(
    message: types.Message,
    document_service: DocumentService,
    settings: Settings,
    rate_limiter: InMemoryRateLimiter,
):
    """
    Handle user document uploads (.pdf, .docx, .txt, .md).
    Downloads file, sends to DocumentService -> VectorRepository (FastEmbed -> Qdrant).
    """
    document = message.document
    if not document:
        return

    if not is_authorized(message, settings):
        await message.answer("You are not authorized to use this bot.")
        return
    user_id = message.from_user.id if message.from_user else None
    if user_id is None:
        await message.answer("Unable to identify your account.")
        return
    if not await rate_limiter.allow(
        f"user:{user_id}",
        settings.BOT_RATE_LIMIT_PER_HOUR,
        settings.RATE_LIMIT_WINDOW_SECONDS,
    ):
        await message.answer("Hourly rate limit exceeded. Please try again later.")
        return

    file_name = sanitize_filename(document.file_name or "")
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if not file_name or extension not in {"pdf", "docx", "txt", "md"}:
        await message.answer("Unsupported file type. Send a PDF, DOCX, TXT, or Markdown document.")
        return
    if document.file_size is not None and document.file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        await message.answer("The document exceeds the upload size limit.")
        return

    status_msg = await message.reply(f"Downloading and indexing: {file_name}...")

    try:
        # Download document buffer
        file_io = await message.bot.download(document.file_id)
        if file_io:
            file_bytes = file_io.read()
            if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
                await status_msg.edit_text("The document exceeds the upload size limit.")
                return
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
            await status_msg.edit_text("Unable to retrieve the document. Please try again.")
    except Exception as exc:
        logger.error("Document upload failed (%s)", type(exc).__name__)
        await status_msg.edit_text("Unable to process the document. Please try again later.")


@router.message(F.photo)
async def handle_unsupported_photo(
    message: types.Message,
    settings: Settings,
    rate_limiter: InMemoryRateLimiter,
):
    if not is_authorized(message, settings):
        await message.answer("You are not authorized to use this bot.")
        return

    user_id = message.from_user.id if message.from_user else None
    if user_id is None:
        await message.answer("Unable to identify your account.")
        return

    if not await rate_limiter.allow(
        f"user:{user_id}",
        settings.BOT_RATE_LIMIT_PER_HOUR,
        settings.RATE_LIMIT_WINDOW_SECONDS,
    ):
        await message.answer("Hourly rate limit exceeded. Please try again later.")
        return

    await message.answer(
        "Photos are not supported yet. Please send a PDF, DOCX, TXT, or Markdown document."
    )

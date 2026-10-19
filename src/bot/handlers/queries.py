import logging
from datetime import datetime, timezone

from aiogram import F, Router, types

from src.bot.handlers.base import is_authorized
from src.core.config import Settings
from src.core.rate_limit import InMemoryRateLimiter
from src.domains.rag.pipeline import RAGPipelineService
from src.domains.rag.schemas import QueryRequest, SearchResultChunk

router = Router(name="query_handlers")
logger = logging.getLogger(__name__)
TELEGRAM_MESSAGE_LIMIT = 4096


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    messages = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            messages.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        messages.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    return messages or [""]


def format_source_citation(chunk: SearchResultChunk) -> str:
    metadata = chunk.metadata
    filename = metadata.get("filename")
    document_title = metadata.get("document_title")
    source = document_title or filename or f"Document {chunk.document_id}"
    if document_title and filename:
        source = f"{source} ({filename})"
    place = metadata.get("place")
    if place:
        source = f"{source} — {place}"

    attribution = []
    sender = metadata.get("uploaded_by")
    username = metadata.get("uploaded_by_username")
    if sender:
        if username:
            sender = f"{sender} (@{username.lstrip('@')})"
        attribution.append(f"sent by {sender}")
    else:
        attribution.append("sender not recorded")
    uploaded_at = metadata.get("uploaded_at")
    if uploaded_at:
        try:
            timestamp = datetime.fromisoformat(str(uploaded_at).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            uploaded_at = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            uploaded_at = str(uploaded_at)
        attribution.append(f"received {uploaded_at}")
    else:
        attribution.append("upload time not recorded")
    suffix = f"; {', '.join(attribution)}" if attribution else ""
    return f"- {source}{suffix}"


@router.message(F.text)
async def handle_user_query(
    message: types.Message,
    rag_pipeline: RAGPipelineService,
    settings: Settings,
    rate_limiter: InMemoryRateLimiter,
):
    """
    Handle plain text questions or messages.
    Triggers RAGPipelineService: FastEmbed query -> Qdrant search -> FlashRank rerank -> Gemini 3.8 Flash.
    """
    user_query = message.text
    if not user_query or user_query.startswith("/"):
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

    status_msg = await message.reply("Searching knowledge base and formulating answer...")

    try:
        query_request = QueryRequest(query=user_query)
        response = await rag_pipeline.execute_query(query_request)

        # Format sources
        sources_text = ""
        if response.source_chunks:
            sources_text = "\n\nSources:\n" + "\n".join(
                format_source_citation(chunk) for chunk in response.source_chunks
            )

        answer = response.answer.strip() or "I couldn't generate an answer from the available information."
        reply_content = f"{answer}{sources_text}"
        parts = split_message(reply_content)
        await status_msg.edit_text(parts[0])
        for part in parts[1:]:
            await message.answer(part)
    except Exception as exc:
        logger.error("RAG query failed (%s)", type(exc).__name__)
        await status_msg.edit_text("Unable to generate an answer right now. Please try again later.")

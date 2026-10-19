import logging
import re
from datetime import datetime, timezone
from typing import Sequence

from aiogram import F, Router, types

from src.bot.formatting import code_entities_for_values
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
    source = document_title or filename or f"Документ {chunk.document_id}"
    if document_title and filename:
        source = f"{source} ({filename})"
    place = metadata.get("place")
    if place:
        source = f"{source} — {place}"

    citation = [f"• {source}"]
    sender = metadata.get("uploaded_by")
    username = metadata.get("uploaded_by_username")
    if sender:
        if username:
            sender = f"{sender} (@{username.lstrip('@')})"
        citation.append(f"  Надіслав: {sender}")
    else:
        citation.append("  Відправника не збережено")
    uploaded_at = metadata.get("uploaded_at")
    if uploaded_at:
        try:
            timestamp = datetime.fromisoformat(str(uploaded_at).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            uploaded_at = timestamp.astimezone(timezone.utc).strftime("%d.%m.%Y о %H:%M UTC")
        except ValueError:
            uploaded_at = str(uploaded_at)
        citation.append(f"  Отримано: {uploaded_at}")
    else:
        citation.append("  Час отримання не збережено")
    return "\n".join(citation)


def format_source_citations(chunks: Sequence[SearchResultChunk]) -> str:
    seen_documents = set()
    citations = []
    for chunk in chunks:
        metadata = chunk.metadata
        identity = chunk.document_id or "|".join(
            str(metadata.get(key, ""))
            for key in ("document_title", "place", "filename", "uploaded_at")
        )
        if identity in seen_documents:
            continue
        seen_documents.add(identity)
        citations.append(format_source_citation(chunk))
    if not citations:
        return ""
    return "Used sources:\n" + "\n".join(citations)


def format_answer_for_telegram(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "• ", line)
        line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
        line = re.sub(r"(?<!\w)([*_])([^*_]+)\1(?!\w)", r"\2", line)
        line = line.replace("```", "").replace("`", "")
        if not line.strip() and (not lines or not lines[-1]):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


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
        sources_text = format_source_citations(response.source_chunks)
        if sources_text:
            sources_text = "\n\n" + sources_text

        answer = format_answer_for_telegram(response.answer)
        answer = answer or "I couldn't generate an answer from the available information."
        reply_content = f"{answer}{sources_text}"
        parts = split_message(reply_content)
        filenames = [
            str(chunk.metadata["filename"])
            for chunk in response.source_chunks
            if chunk.metadata.get("filename")
        ]
        await status_msg.edit_text(
            parts[0],
            entities=code_entities_for_values(parts[0], filenames),
        )
        for part in parts[1:]:
            await message.answer(
                part,
                entities=code_entities_for_values(part, filenames),
            )
    except Exception as exc:
        logger.error("RAG query failed (%s)", type(exc).__name__)
        await status_msg.edit_text("Unable to generate an answer right now. Please try again later.")

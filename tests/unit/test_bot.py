import asyncio
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.bot.handlers.documents import (
    UPLOAD_CAPTION_INSTRUCTIONS,
    handle_document_upload,
    handle_unsupported_photo,
    parse_document_caption,
)
from src.bot.runner import create_bot
from src.core.config import Settings
from src.core.rate_limit import InMemoryRateLimiter


def test_bot_initialization():
    valid_test_token = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ1234"
    with patch("src.bot.runner.get_document_service") as document_service:
        with patch("src.bot.runner.get_rag_pipeline_service") as rag_pipeline:
            document_service.return_value = object()
            rag_pipeline.return_value = object()
            bot, dp = create_bot(token=valid_test_token)
    assert bot is not None
    assert dp is not None
    assert "document_service" in dp.workflow_data
    assert "rag_pipeline" in dp.workflow_data


def test_bot_explains_that_photos_are_unsupported():
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=123),
        answer=AsyncMock(),
    )
    settings = Settings(_env_file=None)

    asyncio.run(
        handle_unsupported_photo(
            message,
            settings,
            InMemoryRateLimiter(),
        )
    )

    message.answer.assert_awaited_once()
    assert "Photos are not supported yet" in message.answer.await_args.args[0]


def test_parse_document_caption_requires_identity_and_place():
    assert parse_document_caption(
        "Документ: Внески скарбника\nМісце: Табір «Сосновий»"
    ) == {
        "document_title": "Внески скарбника",
        "place": "Табір «Сосновий»",
    }
    assert parse_document_caption("Документ: Внески скарбника") is None


def test_document_without_caption_is_not_downloaded_or_indexed():
    message = SimpleNamespace(
        document=SimpleNamespace(
            file_name="contributions.pdf",
            file_size=100,
            file_id="file-id",
        ),
        from_user=SimpleNamespace(id=123),
        caption=None,
        answer=AsyncMock(),
        reply=AsyncMock(),
    )
    document_service = SimpleNamespace(process_and_index_document=AsyncMock())
    settings = Settings(_env_file=None)

    asyncio.run(
        handle_document_upload(
            message,
            document_service,
            settings,
            InMemoryRateLimiter(),
        )
    )

    message.answer.assert_awaited_once_with(UPLOAD_CAPTION_INSTRUCTIONS)
    message.reply.assert_not_awaited()
    document_service.process_and_index_document.assert_not_awaited()


def test_document_upload_passes_sender_time_and_identity_to_indexer():
    timestamp = datetime(2026, 10, 19, 18, 49, tzinfo=timezone.utc)
    status_message = SimpleNamespace(edit_text=AsyncMock())
    message = SimpleNamespace(
        document=SimpleNamespace(
            file_name="contributions.pdf",
            file_size=14,
            file_id="file-id",
        ),
        from_user=SimpleNamespace(
            id=123,
            full_name="Owl",
            username="owl_scout",
        ),
        caption="Документ: Внески скарбника\nМісце: Табір «Сосновий»",
        date=timestamp,
        bot=SimpleNamespace(download=AsyncMock(return_value=BytesIO(b"document bytes"))),
        answer=AsyncMock(),
        reply=AsyncMock(return_value=status_message),
    )
    document_service = SimpleNamespace(
        process_and_index_document=AsyncMock(
            return_value=SimpleNamespace(
                replaced_previous=True,
                filename="contributions.pdf",
                total_chunks=3,
            )
        )
    )

    asyncio.run(
        handle_document_upload(
            message,
            document_service,
            Settings(_env_file=None),
            InMemoryRateLimiter(),
        )
    )

    document_service.process_and_index_document.assert_awaited_once_with(
        file_content=b"document bytes",
        filename="contributions.pdf",
        document_title="Внески скарбника",
        place="Табір «Сосновий»",
        uploaded_by="Owl",
        uploaded_by_username="owl_scout",
        uploaded_by_id=123,
        uploaded_at=timestamp,
    )
    status_message.edit_text.assert_awaited_once()
    confirmation = status_message.edit_text.await_args.args[0]
    assert "previous version was replaced" in confirmation
    assert "Sender: Owl" in confirmation
    assert "2026-10-19 18:49 UTC" in confirmation

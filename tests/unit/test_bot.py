import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.bot.handlers.documents import handle_unsupported_photo
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

import pytest

from src.bot.handlers.queries import split_message
from src.core.rate_limit import InMemoryRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_only_the_configured_number_of_calls():
    limiter = InMemoryRateLimiter()
    assert await limiter.allow("user", limit=2, window_seconds=60)
    assert await limiter.allow("user", limit=2, window_seconds=60)
    assert not await limiter.allow("user", limit=2, window_seconds=60)


@pytest.mark.asyncio
async def test_hourly_rate_limit_allows_sixty_requests_and_rejects_the_next():
    limiter = InMemoryRateLimiter()

    for _ in range(60):
        assert await limiter.allow("user", limit=60, window_seconds=3600)
    assert not await limiter.allow("user", limit=60, window_seconds=3600)


def test_telegram_message_chunks_respect_limit_and_preserve_text():
    text = "x" * 9000
    parts = split_message(text)
    assert all(len(part) <= 4096 for part in parts)
    assert "".join(parts) == text

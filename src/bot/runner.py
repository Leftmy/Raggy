import asyncio
from aiogram import Bot, Dispatcher
from src.core.config import get_settings
from src.core.logger import setup_logger
from src.core.dependencies import (
    get_document_service,
    get_rag_pipeline_service,
    get_vector_repository,
)
from src.core.rate_limit import InMemoryRateLimiter
from src.bot.handlers import base, documents, queries

logger = setup_logger("telegram_bot")


def create_bot(token: str | None = None) -> tuple[Bot, Dispatcher]:
    settings = get_settings()
    configured_token = settings.TELEGRAM_BOT_TOKEN.get_secret_value() if settings.TELEGRAM_BOT_TOKEN else None
    bot_token = token or configured_token
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN must be configured before creating the bot")
    if settings.APP_ENV == "production" and not settings.TELEGRAM_ALLOWED_USER_IDS:
        raise ValueError("TELEGRAM_ALLOWED_USER_IDS must be configured for the production bot")
    bot = Bot(token=bot_token)
    dp = Dispatcher()

    # Dependency Injection into aiogram workflow context
    dp["document_service"] = get_document_service()
    dp["rag_pipeline"] = get_rag_pipeline_service()
    dp["rate_limiter"] = InMemoryRateLimiter()
    dp["settings"] = settings

    dp.include_router(base.router)
    dp.include_router(documents.router)
    dp.include_router(queries.router)

    return bot, dp


async def main():
    settings = get_settings()
    if not settings.TELEGRAM_BOT_TOKEN:
        if settings.APP_ENV == "production":
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the production bot")
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Bot will not start polling.")
        return
    if settings.APP_ENV == "production" and not settings.TELEGRAM_ALLOWED_USER_IDS:
        raise RuntimeError("TELEGRAM_ALLOWED_USER_IDS is required to run the production bot")

    bot, dp = create_bot()
    logger.info("Starting Telegram bot polling with injected dependencies...")
    try:
        await dp.start_polling(bot, close_bot_session=False)
    finally:
        await bot.session.close()
        await get_vector_repository().close()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from typing import Optional
from aiogram import Bot, Dispatcher
from src.core.config import get_settings
from src.core.logger import setup_logger
from src.core.dependencies import get_document_service, get_rag_pipeline_service
from src.bot.handlers import base, documents, queries

logger = setup_logger("telegram_bot")


def create_bot(token: Optional[str] = None) -> tuple[Bot, Dispatcher]:
    settings = get_settings()
    bot_token = token or settings.TELEGRAM_BOT_TOKEN
    bot = Bot(token=bot_token)
    dp = Dispatcher()

    # Dependency Injection into aiogram workflow context
    dp["document_service"] = get_document_service()
    dp["rag_pipeline"] = get_rag_pipeline_service()

    dp.include_router(base.router)
    dp.include_router(documents.router)
    dp.include_router(queries.router)

    return bot, dp


async def main():
    settings = get_settings()
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Bot will not start polling.")
        return

    bot, dp = create_bot()
    logger.info("Starting Telegram bot polling with injected dependencies...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

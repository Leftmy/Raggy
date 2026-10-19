from aiogram import F, Router, types
from aiogram.filters import Command, CommandStart

from src.core.config import Settings

router = Router(name="base_handlers")


def is_authorized(message: types.Message, settings: Settings) -> bool:
    user_id = message.from_user.id if message.from_user else None
    return not settings.TELEGRAM_ALLOWED_USER_IDS or user_id in settings.TELEGRAM_ALLOWED_USER_IDS


@router.message(CommandStart())
async def handle_start(message: types.Message, settings: Settings):
    """Handle /start command."""
    if not is_authorized(message, settings):
        await message.answer("You are not authorized to use this bot.")
        return
    welcome_text = (
        "Hello! I am your RAG assistant.\n\n"
        "Send me a document (.pdf, .docx, .txt, .md) to add it to the knowledge base.\n"
        "Or simply ask a question to retrieve relevant information.\n"
        "Use /help to see these instructions again."
    )
    await message.answer(welcome_text)


@router.message(Command("help"))
async def handle_help(message: types.Message, settings: Settings):
    if not is_authorized(message, settings):
        await message.answer("You are not authorized to use this bot.")
        return
    await message.answer(
        "Send a .pdf, .docx, .txt, or .md document to index it, "
        "or send a text question to search the knowledge base."
    )


@router.message(F.text.startswith("/"))
async def handle_unknown_command(message: types.Message, settings: Settings):
    if not is_authorized(message, settings):
        await message.answer("You are not authorized to use this bot.")
        return
    await message.answer("Unknown command. Use /help to see available actions.")

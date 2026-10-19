from aiogram import Router, types
from aiogram.filters import CommandStart

router = Router(name="base_handlers")


@router.message(CommandStart())
async def handle_start(message: types.Message):
    """Handle /start command."""
    welcome_text = (
        "Hello! I am your RAG assistant.\n\n"
        "Send me a document (.pdf, .docx, .txt) to add it to the knowledge base.\n"
        "Or simply ask a question to retrieve relevant information."
    )
    await message.answer(welcome_text)

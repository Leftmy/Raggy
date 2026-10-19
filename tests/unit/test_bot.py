import pytest
from src.bot.runner import create_bot


def test_bot_initialization():
    valid_test_token = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ1234"
    bot, dp = create_bot(token=valid_test_token)
    assert bot is not None
    assert dp is not None
    assert "document_service" in dp.workflow_data
    assert "rag_pipeline" in dp.workflow_data

import pytest
from pydantic import SecretStr, ValidationError

from src.core.config import Settings


def test_settings_reject_invalid_chunking_configuration():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, CHUNK_SIZE=100, CHUNK_OVERLAP=100)

    with pytest.raises(ValidationError, match="PARENT_CHUNK_SIZE"):
        Settings(_env_file=None, CHUNK_SIZE=500, PARENT_CHUNK_SIZE=400)


def test_production_settings_require_api_auth_and_llm_secret():
    with pytest.raises(ValidationError, match="API_AUTH_TOKEN"):
        Settings(_env_file=None, APP_ENV="production")

    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            API_AUTH_TOKEN=SecretStr("api-token"),
        )


def test_upload_and_rate_limit_defaults():
    settings = Settings(_env_file=None)

    assert settings.MAX_UPLOAD_SIZE_BYTES == 20 * 1024 * 1024
    assert settings.API_RATE_LIMIT_PER_HOUR == 60
    assert settings.BOT_RATE_LIMIT_PER_HOUR == 60
    assert settings.RATE_LIMIT_WINDOW_SECONDS == 60 * 60
    assert settings.PARENT_CHUNK_SIZE == 2000

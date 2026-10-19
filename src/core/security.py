import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.core.config import get_settings

bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    settings = get_settings()
    expected_token = settings.API_AUTH_TOKEN
    if expected_token is None:
        if settings.APP_ENV == "production":
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return

    if (
        credentials is None
        or not secrets.compare_digest(
            credentials.credentials,
            expected_token.get_secret_value(),
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

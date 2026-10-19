import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import get_settings
from src.core.dependencies import (
    get_embedding_service,
    get_llm_service,
    get_reranker_service,
    get_splitter_service,
    get_vector_repository,
)
from src.core.exceptions import (
    AppException,
    DocumentProcessingError,
    EmbeddingError,
    LLMGenerationError,
    RerankerError,
    VectorStoreError,
)
from src.core.logger import setup_logger
from src.core.rate_limit import InMemoryRateLimiter
from src.domains.documents.controller import router as documents_router
from src.domains.rag.controller import router as rag_router


class ApiRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: InMemoryRateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request, call_next):
        settings = get_settings()
        if request.url.path.startswith(settings.API_V1_STR):
            client_host = request.client.host if request.client else "unknown"
            allowed = await self.limiter.allow(
                client_host,
                settings.API_RATE_LIMIT_PER_HOUR,
                settings.RATE_LIMIT_WINDOW_SECONDS,
            )
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS)},
                )
        return await call_next(request)


def create_application() -> FastAPI:
    settings = get_settings()
    setup_logger("rag_app", settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        get_embedding_service()
        get_llm_service()
        get_reranker_service()
        get_splitter_service()
        get_vector_repository()
        yield
        repository = get_vector_repository()
        await repository.close()

    app = FastAPI(
        title="RAG System API",
        version="0.1.0",
        description="RAG API powered by FastEmbed, FlashRank, Qdrant & Gemini 3.8 Flash",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(ApiRateLimitMiddleware, limiter=InMemoryRateLimiter())

    @app.exception_handler(DocumentProcessingError)
    async def document_error_handler(_, exc: DocumentProcessingError):
        return JSONResponse(status_code=400, content={"detail": exc.message})

    @app.exception_handler(VectorStoreError)
    async def vector_store_error_handler(_, exc: VectorStoreError):
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(LLMGenerationError)
    async def llm_error_handler(_, exc: LLMGenerationError):
        return JSONResponse(status_code=502, content={"detail": exc.message})

    @app.exception_handler(EmbeddingError)
    @app.exception_handler(RerankerError)
    async def inference_error_handler(_, exc: AppException):
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(AppException)
    async def app_error_handler(_, exc: AppException):
        return JSONResponse(status_code=500, content={"detail": exc.message})

    app.include_router(documents_router, prefix=settings.API_V1_STR)
    app.include_router(rag_router, prefix=settings.API_V1_STR)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "app_env": settings.APP_ENV}

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        try:
            await get_vector_repository().get_client().get_collections()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Readiness check failed (%s)", type(exc).__name__
            )
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready"}

    return app


app = create_application()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import get_settings
from src.domains.documents.controller import router as documents_router
from src.domains.rag.controller import router as rag_router


def create_application() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG System API",
        version="0.1.0",
        description="RAG API powered by FastEmbed, FlashRank, Qdrant & Gemini 1.5 Pro",
        debug=settings.DEBUG,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents_router, prefix=settings.API_V1_STR)
    app.include_router(rag_router, prefix=settings.API_V1_STR)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "app_env": settings.APP_ENV}

    return app


app = create_application()

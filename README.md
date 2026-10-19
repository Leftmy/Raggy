# RAG Telegram System

A domain-driven Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Aiogram (Telegram Bot)**, **Qdrant**, **FastEmbed**, **FlashRank**, and **Google Gemini 1.5 Pro**.

---

## Architecture (Domain-Driven Design + Repository/Service/Controller)

- **`src/domains/`**: Domain modules containing business logic and abstractions:
  - `documents`: Document lifecycle management (.pdf, .docx, .txt, .md), parsing, text splitting, schemas, and repository contracts.
  - `rag`: End-to-end RAG pipeline (dense query embedding -> vector retrieval -> FlashRank reranking -> LLM answer synthesis).
- **`src/infrastructure/`**: Concrete vendor adapters:
  - `qdrant`: Qdrant vector database repository with `AsyncQdrantClient`.
  - `embeddings`: FastEmbed integration for lightweight CPU-based embeddings.
  - `rerankers`: FlashRank integration for ultra-fast candidate reranking.
  - `llm`: Gemini 1.5 Pro integration via LangChain.
  - `splitters`: Text splitting with recursive character chunking.
- **`src/bot/`**: Telegram bot interface powered by `aiogram` 3.x with dependency injection.
- **`tests/`**: Unit and integration test suite following Test-Driven Development (TDD).

---

## Quick Start

### 1. Environment Setup
Copy the example environment configuration file:
```bash
cp .env.example .env
```
For local development, set the credentials you intend to use in `.env`:
- `TELEGRAM_BOT_TOKEN`
- `LLM_API_KEY`

Configure `API_AUTH_TOKEN` to protect API endpoints. In development it may be omitted; in production it is required. When running the Telegram bot in production, set `TELEGRAM_ALLOWED_USER_IDS` to a JSON array of Telegram numeric user IDs (for example `[123456789]`). An empty list permits all users only outside production.

Uploads are limited by `MAX_UPLOAD_SIZE_BYTES` (20 MiB by default). Accepted formats are PDF, DOCX, TXT, and Markdown; legacy `.doc` is not supported. Telegram photos are not processed; send supported documents as files.

The API and Telegram bot each allow 60 requests per client/user per hour by default. The bot shares a user's hourly quota across text queries and document uploads. Limits are process-local and require a shared backend such as Redis when running multiple replicas.

### 2. Run with Docker Compose
```bash
docker compose up --build
```
This starts:
- **Vector DB (Qdrant)** on the internal Compose network only.
- **FastAPI backend** on `127.0.0.1:8000` (API documentation available at `http://localhost:8000/docs`).
- **Telegram Bot** polling service.

### 3. Run Tests (TDD)
```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

To run the same suite in the Python 3.11 test image:

```bash
docker build --target tester -t rag-system-tests .
docker run --rm rag-system-tests
```

`requirements.txt` installs runtime dependencies declared in `pyproject.toml`; install `requirements-dev.txt` to add test tools.

### API access

When `API_AUTH_TOKEN` is set, send it as a Bearer token to `/api/v1` endpoints:

```text
Authorization: Bearer <API_AUTH_TOKEN>
```

`/health` is a liveness check. `/ready` checks connectivity to Qdrant and is used by Compose for readiness.

### Operations

- Restrict CORS through the JSON-array `CORS_ORIGINS` setting.
- API and bot rate limits are process-local; use a shared limiter such as Redis when running multiple replicas.
- Compose does not publish Qdrant ports to the host. Use the Docker network or a separately secured administrative channel.

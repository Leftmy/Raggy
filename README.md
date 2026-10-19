# RAG Telegram System

A domain-driven Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Aiogram (Telegram Bot)**, **Qdrant**, **FastEmbed**, **FlashRank**, and **Google Gemini 1.5 Pro**.

---

## Architecture (Domain-Driven Design + Repository/Service/Controller)

- **`src/domains/`**: Domain modules containing business logic and abstractions:
  - `documents`: Document lifecycle management (.pdf, .docx, .txt), parsing, text splitting, schemas, and repository contracts.
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
Fill in the required credentials in `.env`:
- `TELEGRAM_BOT_TOKEN`
- `LLM_API_KEY`

### 2. Run with Docker Compose
```bash
docker-compose up --build
```
This starts:
- **Vector DB (Qdrant)** on ports `6333` (HTTP) and `6334` (gRPC).
- **FastAPI backend** on port `8000` (API documentation available at `http://localhost:8000/docs`).
- **Telegram Bot** polling service.

### 3. Run Tests (TDD)
```bash
docker run --rm -e PYTHONPATH=/app rag-app-test pytest -v
```
or run locally:
```bash
pytest -v
```

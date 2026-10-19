# AGENTS.md — System Architecture & Decisions Log

## 1. Project Overview & Mission

This project is a production-ready, domain-driven **Retrieval-Augmented Generation (RAG)** system integrated with a **Telegram Bot** and a **FastAPI** backend.

The system ingests and processes multiple document formats (`.pdf`, `.docx`, `.txt`) as well as direct text inputs, generates vector embeddings, stores and searches them in a vector database, reranks candidate chunks for optimal relevance, and generates contextual answers using Google Gemini 1.5 Pro.

---

## 2. Core Architectural & Technology Decisions

### 2.1 Technology Stack
- **Language & Runtime:** Python 3.11+
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) with Uvicorn ASGI server.
- **Telegram Bot Framework:** [aiogram 3.x](https://docs.aiogram.dev/) (modern asynchronous framework).
- **Vector Database:** [Qdrant](https://qdrant.tech/) (scalable vector search engine with gRPC and HTTP APIs).
- **Embedding Generation:** [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5` by default) — chosen for lightweight, fast, local CPU-optimized embeddings without heavy PyTorch dependencies.
- **Reranker:** [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (`ms-marco-TinyBERT-L-2-v2`) — fast, Ultra-lite re-ranking engine designed for low-latency reranking before LLM context construction.
- **LLM & Orchestration:** [LangChain](https://www.langchain.com/) / `langchain-google-genai` powering Google Gemini 1.5 Pro.
- **Document Parsers:** `pypdf` (for PDF files), `python-docx` (for Word documents), and native Python UTF-8 text decoders.

---

## 3. Structural Patterns & Design Principles

### 3.1 Domain-Driven Design (DDD) & Layered Architecture
The codebase separates domains logically rather than dumping all models, controllers, and services into flat directories:
- **`src/domains/documents`**: Handles document lifecycle — parsing raw streams, text splitting/chunking, document schemas, and vector repository abstraction contracts.
- **`src/domains/rag`**: Manages query embedding, vector retrieval, candidate reranking, prompt augmentation, and response synthesis.
- **`src/infrastructure/`**: Concrete implementations of external services:
  - `qdrant/`: Client interaction and Qdrant repository implementation.
  - `embeddings/`: FastEmbed wrapper implementing the domain `EmbeddingService` interface.
  - `rerankers/`: FlashRank wrapper implementing the domain `RerankerService` interface.
  - `llm/`: Gemini 1.5 Pro wrapper implementing the domain `LLMService` interface.
- **`src/bot/`**: Telegram bot presentation layer decoupled from backend internals, communicating directly with domain services.

### 3.2 Architectural Pattern: Repository – Service – Controller
- **Controllers / Routers:** Parse incoming HTTP or Telegram events, perform request validation, and delegate to domain services.
- **Services:** Contain pure business logic and orchestration (e.g. `RAGPipelineService`, `DocumentService`).
- **Repositories:** Abstract database and external vector storage operations behind interfaces (`VectorRepository`), decoupling the core domain from the concrete database driver.

### 3.3 Dependency Inversion & Inversion of Control (DI)
To eliminate tight coupling with specific external vendors (Gemini, Qdrant, FastEmbed, FlashRank, LangChain):
- **Abstract Interfaces in Core/Domains:**
  - `VectorRepository` (in `src/domains/documents/repositories.py`) abstracts the vector database.
  - `EmbeddingService` (in `src/domains/rag/service.py`) abstracts dense embedding generation.
  - `RerankerService` (in `src/domains/rag/service.py`) abstracts candidate reranking.
  - `LLMService` (in `src/domains/rag/service.py`) abstracts generative answer synthesis.
  - `TextSplitterService` (in `src/domains/documents/splitters.py`) abstracts document chunking.
- **Centralized DI Factory (`src/core/dependencies.py`):**
  - Instantiates concrete implementations (`QdrantVectorRepository`, `FastEmbedService`, `FlashRankReranker`, `GeminiLLMService`, `LangChainRecursiveSplitter`).
  - Injected into FastAPI routes via `Depends(...)`.
  - Injected into Aiogram bot handlers via `Dispatcher` workflow context (`dp["document_service"]`, `dp["rag_pipeline"]`).
  - Enables zero-friction replacement of any vendor (e.g. swapping Qdrant for Pinecone/Milvus, FastEmbed for OpenAI, or Gemini for Claude/Local LLMs) and seamless test mocking via `app.dependency_overrides`.

### 3.4 Test-Driven Development (TDD)
- The test suite is organized into `tests/unit` and `tests/integration`.
- Test contracts are laid out first to guide implementation:
  - Unit tests verify document parsing and RAG pipeline coordination using mocked interfaces.
  - Integration tests verify API endpoints and health checks using `httpx.AsyncClient` and dependency overrides.

---

## 4. End-to-End Pipelines

### 4.1 Ingestion Pipeline
```
User / API / Telegram Doc (.pdf, .docx, .txt)
                    │
                    ▼
     [DocumentParser Factory]
                    │
                    ▼
     [Text Splitter / Chunker]
                    │
                    ▼
         [FastEmbed Service]
                    │
                    ▼
  [QdrantVectorRepository.save_chunks]
```

### 4.2 Retrieval & Generation (RAG) Pipeline
```
User Query (Telegram or API)
                    │
                    ▼
         [FastEmbed Service]
         (Query Vectorization)
                    │
                    ▼
          [Qdrant Vector DB]
   (Retrieve Top-K Candidates, e.g., K=10)
                    │
                    ▼
         [FlashRank Reranker]
    (Rerank Candidates to Top-N, e.g., N=3)
                    │
                    ▼
         [Prompt Construction]
   (Context Chunks + User Question)
                    │
                    ▼
       [Google Gemini 1.5 Pro]
                    │
                    ▼
  Final Formatted Answer returned to User
```

---

## 5. Configuration & Environment Variables

All configurations are strictly managed through technology-agnostic environment variables via `pydantic-settings` in `src/core/config.py`:
- `TELEGRAM_BOT_TOKEN`: Telegram bot authentication token from @BotFather.
- `LLM_PROVIDER`: Pluggable LLM provider (`gemini`, `openai`, `anthropic`, etc., default: `gemini`).
- `LLM_API_KEY`: API authentication key for the selected LLM provider.
- `LLM_MODEL`: Target model identifier (default: `gemini-1.5-pro`).
- `VECTOR_STORE_PROVIDER`: Pluggable vector store (`qdrant`, `pinecone`, `milvus`, etc., default: `qdrant`).
- `VECTOR_STORE_HOST`, `VECTOR_STORE_PORT`, `VECTOR_STORE_COLLECTION`: Target vector store instance & collection.
- `EMBEDDING_PROVIDER`: Pluggable embedding service (`fastembed`, `openai`, etc., default: `fastembed`).
- `EMBEDDING_MODEL`: Embedding model identifier (default: `BAAI/bge-small-en-v1.5`).
- `RERANKER_PROVIDER`: Pluggable reranking service (`flashrank`, `cohere`, etc., default: `flashrank`).
- `RERANKER_MODEL`: Reranking model identifier (default: `ms-marco-TinyBERT-L-2-v2`).
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RETRIEVAL`, `TOP_K_RERANK`: Tuning parameters for chunking and search accuracy.

---

## 6. Containerization & Deployment Setup

The system is configured for multi-container orchestration via `docker-compose.yaml`:
1. **`vector_db`**: Vector database container running on ports `6333` (HTTP) and `6334` (gRPC) with persistent volume storage (`vector_db_storage`).
2. **`api`**: FastAPI service running Uvicorn with hot-reload enabled for development.
3. **`bot`**: Dedicated container running the aiogram Telegram polling bot (`src.bot.runner`).

---

## 7. Guidelines for Future Agentic & Developer Contributions

When continuing development:
1. **Follow TDD:** Write or expand unit tests in `tests/` before implementing concrete method bodies.
2. **Preserve Domain Boundaries:** Do not import `src/infrastructure/` directly into controllers; inject services conforming to abstract interfaces defined in `src/domains/`.
3. **Environment Security:** Never commit real secrets (`.env`); always update `.env.example` when adding new configuration keys.

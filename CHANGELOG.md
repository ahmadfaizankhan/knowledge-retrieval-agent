# Changelog

All notable changes to the Knowledge Retrieval Agent are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-06-15

Initial production-ready release of the end-to-end RAG pipeline.

### Added

**Ingestion (Phase 1)**
- `DocumentLoaderFactory` with format-specific loaders for PDF, TXT, MD, DOCX and
  an `UnstructuredFileLoader` fallback; bulk directory loading with fail-forward
  error handling.
- `ChunkingEngine` using `RecursiveCharacterTextSplitter` (configurable size /
  overlap), optional `SemanticChunker`, short-chunk filtering, and per-chunk
  metadata enrichment (`chunk_index`, `ingestion_timestamp`, `chunk_hash`, UUID
  `chunk_id`, `page_number`).
- SHA-256 deduplication backed by a SQLite metadata store, with a `--force`
  override.
- `IngestionPipeline` orchestrator (`load → split → filter → dedup → embed →
  upsert`) with a structured JSON completion event and CLI.

**Embeddings & vector stores (Phase 1–2)**
- `EmbeddingFactory` with pluggable `openai` / `huggingface` / `local` backends;
  OpenAI calls wrapped in `tenacity` exponential-backoff retry.
- Deterministic, dependency-free `local` embeddings (presence-based hashing,
  2048-dim) enabling fully offline operation.
- `VectorStoreManager` for ChromaDB (cosine space, dev) and Pinecone Serverless
  (batched upserts, index verification/creation, prod), with store caching +
  locking for safe concurrent access.

**Retrieval & generation (Phase 2)**
- `RetrieverFactory` (MMR + similarity), metadata-filter injection.
- Optional `CrossEncoderReranker` (guarded by `ENABLE_RERANKER`, graceful no-op
  fallback when `sentence-transformers` is absent).
- `LLMFactory` with `openai` (ChatOpenAI, temp 0.0, max 1024) and `local`
  extractive answerer honouring the PRD guardrail prompt.
- `RAGChainBuilder` (LangChain `RetrievalQAWithSourcesChain` /
  `ConversationalRetrievalChain`) and `RAGService` (scored retrieval, threshold
  fallback that skips the LLM when no context passes, content dedup, structured
  citation-backed responses).

**API & workflows (Phase 3)**
- FastAPI service: `/health`, `/query`, `/ingest`, `/metrics`, with `X-API-Key`
  auth, structured request/response logging and query audit logging.
- Pydantic schemas matching PRD §2.5 (`QueryResponse`, `Source`, etc.).
- n8n `doc_ingestion_workflow.json` (8 nodes) and `query_routing_workflow.json`
  (6 nodes) with retry + error branches.

**Observability (Phase 4)**
- `structlog` JSON logging (stdout + rotating file) with the required fields.
- Prometheus metrics: `query_latency_seconds`, `ingestion_chunks_total`,
  `retrieval_score_mean`, `llm_tokens_used_total`, `queries_total`.
- Grafana dashboard (`monitoring/grafana_dashboard.json`).

**Testing & evaluation (Phase 4)**
- 42 unit + integration tests (81% coverage), hermetic and offline.
- RAGAS evaluation harness with a 20-item ground-truth dataset and a
  deterministic offline-fallback evaluator.
- Retrieval/ingestion benchmark script (latency percentiles, throughput).

**Tooling**
- `Dockerfile` (python:3.11-slim) + `docker-compose.yml` (ChromaDB + API),
  `Makefile`, `pyproject.toml`, `.env.example`, `.gitignore`, README and RUNBOOK.

### Configuration decisions
- Defaults ship **fully offline** (`local` embeddings + `local` LLM + Chroma) so
  the system runs with zero API keys; production semantic quality is enabled by
  switching `EMBEDDING_PROVIDER`/`LLM_PROVIDER`/`VECTOR_STORE`.
- The score threshold is provider-aware: the PRD's `0.72` applies to semantic
  (OpenAI/HF) embeddings; the offline lexical backend uses a relaxed floor.

### Known limitations
- Pinecone, `ragas`, `sentence-transformers` and `unstructured` are not installed
  in the local Python 3.14 dev environment (no compatible wheels yet); their code
  paths are lazily imported and exercised in the Python 3.11 Docker image.
- The offline `local` embedding is a lexical stand-in; RAGAS quality targets are
  validated on the OpenAI path (see README).

### Evaluation results (offline-fallback, sample corpus)
- Faithfulness 1.00 · Context Recall 0.99 · Context Precision 0.94 ·
  Hallucination rate 0.00 · Answer Relevancy 0.47 (limited by the lexical
  offline embedding; validate relevancy on the OpenAI + RAGAS path).

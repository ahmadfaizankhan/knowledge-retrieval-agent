# Knowledge Retrieval Agent

Knowledge Retrieval Agent is a FastAPI service for turning internal documents into a searchable question-answering system. It ingests PDFs, Word documents, Markdown, and text files, splits them into traceable chunks, stores embeddings in a vector database, and answers questions with source references.

The project is designed to run locally without external keys for development, while still supporting OpenAI and Pinecone for production deployments.

## Features

- Document ingestion for `.pdf`, `.docx`, `.md`, and `.txt` files
- Chunking, deduplication, and metadata tracking
- Local development mode with ChromaDB and deterministic local providers
- Production-ready provider options for OpenAI embeddings, OpenAI chat models, and Pinecone
- FastAPI endpoints for health checks, document ingestion, querying, and Prometheus metrics
- Structured JSON logging and a SQLite metadata store for ingestion/query audit records
- n8n workflow examples for ingestion and query routing
- Unit, integration, evaluation, and benchmarking scripts

## Project Structure

```text
api/                 FastAPI application, request models, and dependencies
config/              Environment-based application settings
core/                Logging, metrics, metadata storage, and shared exceptions
embeddings/          Embedding provider factory and vector-store upsert logic
ingestion/           Document loading, splitting, and ingestion pipeline
retrieval/           Retriever, reranker, LLM wrapper, and query service
scripts/             Utility scripts for seeding, benchmarking, and Pinecone checks
tests/               Unit, integration, fixture, and evaluation files
n8n/workflows/       Example workflow definitions
monitoring/          Grafana dashboard configuration
docs/                Runbook and setup notes
```

## Requirements

- Python 3.11 or newer
- pip
- Docker and Docker Compose, if you want to run the containerized stack

OpenAI and Pinecone keys are optional for local development. The default configuration uses local providers and ChromaDB.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

The default values in `.env.example` are suitable for local development. Keep real secrets in `.env`; it is intentionally ignored by Git.

## Run Locally

Seed the local Chroma database with the sample documents:

```bash
python scripts/seed_chroma.py --dir tests/fixtures/sample_docs --namespace test --force
```

Start the API:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Check the service:

```bash
curl http://localhost:8080/health
```

Ask a question:

```bash
curl -X POST http://localhost:8080/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is the refund window?\", \"namespace\": \"test\"}"
```

For macOS or Linux shells, use line continuations with `\` instead of `^`.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Reports API, vector store, and model-provider status |
| `POST` | `/query` | Answers a question from indexed content and returns sources |
| `POST` | `/ingest` | Ingests a file path, directory, or base64 file payload |
| `GET` | `/metrics` | Exposes Prometheus metrics |

When `REQUIRE_API_KEY=true`, protected endpoints require an `X-API-Key` header matching `FASTAPI_API_KEY`.

## Docker

Create `.env` first, then start the API and ChromaDB:

```bash
docker compose up -d
```

The API will be available at:

```text
http://localhost:8080
```

## Production Configuration

For a production-style deployment, update `.env` with real provider settings:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
VECTOR_STORE=pinecone
REQUIRE_API_KEY=true
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
FASTAPI_API_KEY=your-service-api-key
```

Verify or create the Pinecone index:

```bash
python scripts/verify_pinecone.py --create-if-missing
```

Ingest production documents:

```bash
python -m ingestion.pipeline --dir ./docs --vector-store pinecone --namespace production
```

## Testing

Run unit tests:

```bash
python -m pytest tests/unit -v
```

Run integration tests against Chroma:

```bash
python -m pytest tests/integration --vector-store chroma -v
```

Run the evaluation script:

```bash
python tests/eval/evaluate_hallucination.py --seed
```

Run the retrieval benchmark:

```bash
python scripts/benchmark_retrieval.py --queries tests/eval/rag_eval_dataset.json --n 100 --concurrency 10
```

## Security Notes

- Do not commit `.env`, API keys, local databases, logs, vector indexes, or generated evaluation reports.
- Use `REQUIRE_API_KEY=true` outside local development.
- Rotate keys immediately if a secret is ever exposed in logs, screenshots, or commit history.
- Review `.env.example` before sharing the repository to make sure it contains placeholders only.
- Treat Chroma, SQLite, and log files as runtime data; they can contain private source text or user queries.

## License

No license has been provided yet. Add one before publishing if you want others to use or contribute to the project.

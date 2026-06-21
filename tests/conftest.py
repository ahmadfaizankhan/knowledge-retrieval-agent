"""Shared pytest fixtures.

Forces the fully-offline configuration (local embeddings + local LLM + Chroma)
and isolates the vector store / metadata DB to a temp directory so the suite is
hermetic and requires no API keys.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- Configure environment BEFORE any app module reads settings -------------
_TMP = tempfile.mkdtemp(prefix="kra_tests_")
os.environ.update(
    {
        "EMBEDDING_PROVIDER": "local",
        "LLM_PROVIDER": "local",
        "VECTOR_STORE": "chroma",
        "CHROMA_PERSIST_DIR": str(Path(_TMP) / "chroma_db"),
        "CHROMA_COLLECTION": "knowledge_base_dev",
        "METADATA_DB_URL": f"sqlite:///{Path(_TMP) / 'metadata.sqlite3'}",
        "LOG_DIR": str(Path(_TMP) / "logs"),
        "REQUIRE_API_KEY": "false",
        "FASTAPI_API_KEY": "test-key",
        "SCORE_THRESHOLD": "0.2",
        "RETRIEVAL_STRATEGY": "similarity",
    }
)

# Clear any cached settings so the env above takes effect.
from config.settings import get_settings  # noqa: E402

get_settings.cache_clear()

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_DOCS = FIXTURES / "sample_docs"


def pytest_addoption(parser):
    parser.addoption(
        "--vector-store",
        action="store",
        default="chroma",
        choices=["chroma", "pinecone"],
        help="Vector store to run integration tests against.",
    )


@pytest.fixture(scope="session")
def vector_store(request) -> str:
    return request.config.getoption("--vector-store")


@pytest.fixture(scope="session")
def sample_docs_dir() -> str:
    return str(SAMPLE_DOCS)


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture()
def seeded_namespace(sample_docs_dir):
    """Ingest the sample docs once and return the namespace used."""
    from ingestion.pipeline import IngestionPipeline

    ns = "test"
    pipeline = IngestionPipeline()
    event = pipeline.run(
        path=sample_docs_dir,
        is_directory=True,
        namespace=ns,
        vector_store="chroma",
        force=True,
    )
    assert event["chunks_upserted"] > 0
    return ns

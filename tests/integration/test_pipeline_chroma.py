"""Integration tests for the full ingestion pipeline against ChromaDB."""

from __future__ import annotations

import pytest

from embeddings.upsert import VectorStoreManager
from ingestion.pipeline import IngestionPipeline


@pytest.fixture()
def fresh_pipeline(settings):
    return IngestionPipeline(settings)


def test_pipeline_ingests_directory(fresh_pipeline, sample_docs_dir):
    event = fresh_pipeline.run(
        path=sample_docs_dir,
        is_directory=True,
        namespace="test",
        vector_store="chroma",
        force=True,
    )
    assert event["status"] == "complete"
    assert event["chunks_upserted"] > 0
    assert event["errors"] == []


def test_pipeline_dedup_on_second_run(fresh_pipeline, sample_docs_dir):
    # First run with force seeds the data.
    fresh_pipeline.run(
        path=sample_docs_dir, is_directory=True, namespace="dedup",
        vector_store="chroma", force=True,
    )
    # Second run WITHOUT force should skip everything as duplicates.
    second = fresh_pipeline.run(
        path=sample_docs_dir, is_directory=True, namespace="dedup",
        vector_store="chroma", force=False,
    )
    assert second["chunks_upserted"] == 0
    assert second["chunks_skipped"] > 0


def test_chroma_collection_count_matches(fresh_pipeline, sample_docs_dir):
    fresh_pipeline.run(
        path=sample_docs_dir, is_directory=True, namespace="count",
        vector_store="chroma", force=True,
    )
    vsm = VectorStoreManager(fresh_pipeline.settings)
    store = vsm.get_chroma_store()
    assert store._collection.count() > 0


def test_single_file_ingestion(fresh_pipeline, sample_docs_dir):
    event = fresh_pipeline.run(
        path=f"{sample_docs_dir}/financial_report.md",
        is_directory=False,
        namespace="single",
        vector_store="chroma",
        force=True,
    )
    assert event["chunks_upserted"] > 0

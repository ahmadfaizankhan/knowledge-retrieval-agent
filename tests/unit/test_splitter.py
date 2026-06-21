"""Unit tests for the chunking engine."""

from __future__ import annotations

from langchain_core.documents import Document

from ingestion.splitter import ChunkingEngine, compute_chunk_hash


def _make_docs():
    text = " ".join(f"word{i}" for i in range(500))
    return [Document(page_content=text, metadata={"source_file": "x.txt", "doc_type": "txt"})]


def test_split_recursive_produces_chunks(settings):
    engine = ChunkingEngine(settings)
    chunks = engine.split_recursive(_make_docs(), chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == idx
        assert "ingestion_timestamp" in c.metadata
        assert "chunk_hash" in c.metadata
        assert "chunk_id" in c.metadata


def test_chunk_hash_is_deterministic():
    assert compute_chunk_hash("hello") == compute_chunk_hash("hello")
    assert compute_chunk_hash("hello") != compute_chunk_hash("world")


def test_filter_short_chunks(settings):
    engine = ChunkingEngine(settings)
    docs = [
        Document(page_content="too short", metadata={}),
        Document(page_content=" ".join(["w"] * 50), metadata={}),
    ]
    kept = engine.filter_short_chunks(docs, min_words=20)
    assert len(kept) == 1
    assert len(kept[0].page_content.split()) >= 20


def test_all_chunks_meet_min_words(settings, sample_docs_dir):
    from ingestion.loader import DocumentLoaderFactory

    engine = ChunkingEngine(settings)
    docs = DocumentLoaderFactory.load_directory(sample_docs_dir)
    chunks = engine.filter_short_chunks(engine.split_recursive(docs))
    assert chunks
    assert all(len(c.page_content.split()) >= settings.min_chunk_words for c in chunks)

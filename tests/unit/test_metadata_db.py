"""Unit tests for the SQLite metadata DB (dedup + audit)."""

from __future__ import annotations

from core.metadata_db import MetadataDB


def test_hash_dedup(tmp_path):
    db = MetadataDB(url=f"sqlite:///{tmp_path / 'm.sqlite3'}")
    assert not db.hash_exists("abc")
    db.record_hash("abc", "id1", "f.txt", "ns")
    assert db.hash_exists("abc")
    # Idempotent insert does not raise.
    db.record_hash("abc", "id1", "f.txt", "ns")


def test_ingestion_and_query_logs(tmp_path):
    db = MetadataDB(url=f"sqlite:///{tmp_path / 'm.sqlite3'}")
    db.log_ingestion(
        request_id="r1", doc_name="d", namespace="ns",
        chunks_total=10, chunks_upserted=8, chunks_skipped=2,
        tokens_estimated=100, duration_ms=50, errors=[],
    )
    db.log_query(
        request_id="q1", query="hi", namespace="ns", answer="a",
        confidence_score=0.9, latency_ms=42, sources=["c1"],
    )
    recent = db.recent_queries(limit=5)
    assert recent and recent[0]["query"] == "hi"

"""Lightweight metadata store backed by SQLite (FR-ING-003, FR-N8N-003).

Tracks three things:

* ``chunk_hashes``  — SHA-256 of every upserted chunk, for deduplication.
* ``ingestion_log`` — one row per ingestion run (observability / audit).
* ``query_log``     — one row per served query (audit trail).

The default ``sqlite:///`` URL is zero-config and perfect for local dev / CI.
A Postgres URL can be supplied in production; the schema is portable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import get_settings


def _sqlite_path_from_url(url: str) -> str:
    """Extract a filesystem path from a ``sqlite:///path`` URL."""
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///") :]
    if url.startswith("sqlite://"):
        return url[len("sqlite://") :]
    # Fall back to treating the value as a bare path.
    return url


class MetadataDB:
    """Thread-safe SQLite wrapper for dedup + audit logging."""

    def __init__(self, url: str | None = None):
        settings = get_settings()
        self.url = url or settings.metadata_db_url
        if not self.url.startswith("sqlite"):
            raise ValueError(
                "MetadataDB currently supports sqlite URLs only. "
                f"Got: {self.url!r}. Use sqlite:///./metadata.sqlite3"
            )
        self.path = _sqlite_path_from_url(self.url)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunk_hashes (
                    chunk_hash   TEXT PRIMARY KEY,
                    chunk_id     TEXT NOT NULL,
                    source_file  TEXT,
                    namespace    TEXT,
                    created_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id       TEXT,
                    doc_name         TEXT,
                    namespace        TEXT,
                    chunks_total     INTEGER,
                    chunks_upserted  INTEGER,
                    chunks_skipped   INTEGER,
                    tokens_estimated INTEGER,
                    duration_ms      INTEGER,
                    errors           TEXT,
                    created_at       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id       TEXT,
                    query            TEXT,
                    namespace        TEXT,
                    answer           TEXT,
                    confidence_score REAL,
                    latency_ms       INTEGER,
                    sources          TEXT,
                    created_at       TEXT NOT NULL
                );
                """
            )

    # --- Deduplication ------------------------------------------------------
    def hash_exists(self, chunk_hash: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM chunk_hashes WHERE chunk_hash = ? LIMIT 1",
                (chunk_hash,),
            )
            return cur.fetchone() is not None

    def record_hash(
        self,
        chunk_hash: str,
        chunk_id: str,
        source_file: str | None,
        namespace: str | None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO chunk_hashes "
                "(chunk_hash, chunk_id, source_file, namespace, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk_hash, chunk_id, source_file, namespace, _now()),
            )

    # --- Ingestion audit ----------------------------------------------------
    def log_ingestion(self, **fields: Any) -> None:
        fields.setdefault("created_at", _now())
        if isinstance(fields.get("errors"), (list, dict)):
            fields["errors"] = json.dumps(fields["errors"])
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO ingestion_log ({cols}) VALUES ({placeholders})",
                tuple(fields.values()),
            )

    # --- Query audit --------------------------------------------------------
    def log_query(self, **fields: Any) -> None:
        fields.setdefault("created_at", _now())
        if isinstance(fields.get("sources"), (list, dict)):
            fields["sources"] = json.dumps(fields["sources"])
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO query_log ({cols}) VALUES ({placeholders})",
                tuple(fields.values()),
            )

    def recent_queries(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT query, confidence_score, latency_ms, created_at "
                "FROM query_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_DB: MetadataDB | None = None


def get_metadata_db() -> MetadataDB:
    """Return a process-wide singleton MetadataDB."""
    global _DB
    if _DB is None:
        _DB = MetadataDB()
    return _DB

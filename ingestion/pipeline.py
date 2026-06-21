"""End-to-end ingestion pipeline orchestrator (FR-ING-004, Phase 1.5).

Orchestrates: load -> split -> filter -> deduplicate -> embed -> upsert, with
fail-forward error handling, structured logging and a JSON completion event.

CLI:
    python -m ingestion.pipeline --dir docs/ --vector-store chroma --namespace test
    python -m ingestion.pipeline --file report.pdf --vector-store pinecone --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any

from config.settings import Settings, get_settings
from core.exceptions import DocumentLoadError
from core.logging import get_logger
from core.metadata_db import get_metadata_db
from embeddings.embedder import EmbeddingFactory
from embeddings.upsert import VectorStoreManager
from ingestion.loader import DocumentLoaderFactory
from ingestion.splitter import ChunkingEngine

logger = get_logger("ingestion.pipeline")


class IngestionPipeline:
    """Runs documents through the full ingestion flow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embeddings = EmbeddingFactory.create(self.settings)
        self.chunker = ChunkingEngine(self.settings)
        self.vsm = VectorStoreManager(self.settings, embeddings=self.embeddings)
        self.metadata_db = get_metadata_db()

    def run(
        self,
        path: str,
        is_directory: bool,
        namespace: str = "default",
        vector_store: str | None = None,
        force: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        vector_store = vector_store or self.settings.vector_store
        start = time.perf_counter()
        errors: list[dict[str, Any]] = []

        # --- Load ----------------------------------------------------------
        try:
            if is_directory:
                docs = DocumentLoaderFactory.load_directory(path)
            else:
                docs = DocumentLoaderFactory.load(path)
        except DocumentLoadError as exc:
            errors.append({"stage": "load", "error": str(exc)})
            return self._finish(
                request_id, path, namespace, 0, 0, 0, 0, start, errors, status="error"
            )

        # --- Split + filter ------------------------------------------------
        chunks = self.chunker.split(docs, embeddings=self.embeddings)
        chunks = self.chunker.filter_short_chunks(chunks)
        tokens_estimated = sum(len(c.page_content) // 4 for c in chunks)

        # --- Embed + upsert (fail-forward per batch) -----------------------
        upserted = 0
        skipped = 0
        try:
            result = self.vsm.upsert(
                chunks, vector_store=vector_store, namespace=namespace, force=force
            )
            upserted = result["chunks_upserted"]
            skipped = result["chunks_skipped"]
        except Exception as exc:  # noqa: BLE001
            errors.append({"stage": "upsert", "error": repr(exc)})

        return self._finish(
            request_id,
            path,
            namespace,
            len(chunks),
            upserted,
            skipped,
            tokens_estimated,
            start,
            errors,
            status="complete" if not errors else "partial",
        )

    def _finish(
        self,
        request_id,
        path,
        namespace,
        chunks_total,
        upserted,
        skipped,
        tokens_estimated,
        start,
        errors,
        status,
    ) -> dict[str, Any]:
        duration_ms = int((time.perf_counter() - start) * 1000)
        doc_name = path
        event = {
            "status": status,
            "request_id": request_id,
            "doc_name": doc_name,
            "namespace": namespace,
            "chunks_total": chunks_total,
            "chunks_upserted": upserted,
            "chunks_skipped": skipped,
            "tokens_estimated": tokens_estimated,
            "duration_ms": duration_ms,
            "errors": errors,
        }
        self.metadata_db.log_ingestion(
            request_id=request_id,
            doc_name=doc_name,
            namespace=namespace,
            chunks_total=chunks_total,
            chunks_upserted=upserted,
            chunks_skipped=skipped,
            tokens_estimated=tokens_estimated,
            duration_ms=duration_ms,
            errors=errors,
        )
        logger.info("ingestion_complete", **event)
        return event


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge agent ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory of documents to ingest")
    group.add_argument("--file", help="Single document to ingest")
    parser.add_argument("--namespace", default="default")
    parser.add_argument(
        "--vector-store", choices=["chroma", "pinecone"], default=None
    )
    parser.add_argument("--force", action="store_true", help="Bypass deduplication")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pipeline = IngestionPipeline()
    path = args.dir or args.file
    event = pipeline.run(
        path=path,
        is_directory=bool(args.dir),
        namespace=args.namespace,
        vector_store=args.vector_store,
        force=args.force,
    )
    # Emit the structured JSON completion event to stdout (FR-ING-004).
    print(json.dumps(event))
    return 0 if event["status"] in ("complete", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())

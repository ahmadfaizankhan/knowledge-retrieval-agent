"""FastAPI application (Phase 3.1).

Endpoints:
* ``GET  /health``  — subsystem health + vector count (FR-REL-002)
* ``POST /query``   — RAG query, returns structured QueryResponse
* ``POST /ingest``  — trigger ingestion (path / directory / base64)
* ``GET  /metrics`` — Prometheus exposition (FR-REL-004)

All write/query endpoints depend on ``require_api_key`` (FR-SEC-005) and emit
structured request/response logs.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST

from api.deps import get_rag_service, require_api_key
from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SubsystemStatus,
)
from config.settings import get_settings
from core.exceptions import LLMGenerationError
from core.logging import get_logger
from core.metadata_db import get_metadata_db
from core.metrics import QUERIES_TOTAL, QUERY_LATENCY_SECONDS, render_metrics
from ingestion.pipeline import IngestionPipeline

logger = get_logger("api")

app = FastAPI(
    title="Knowledge Retrieval Agent",
    version="1.0.0",
    description="Production RAG pipeline: ingest -> retrieve -> generate.",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    svc = get_rag_service()

    # Vector DB connectivity + count.
    vector_db = SubsystemStatus(status="ok")
    vector_count = 0
    try:
        store = svc.vsm.get_store(namespace=settings.pinecone_namespace)
        try:
            vector_count = store._collection.count()  # Chroma
        except Exception:  # noqa: BLE001 - Pinecone path / unknown store
            vector_count = -1
    except Exception as exc:  # noqa: BLE001
        vector_db = SubsystemStatus(status="error", detail=repr(exc))

    # LLM reachability (local is always reachable; openai requires a key).
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        llm_status = SubsystemStatus(status="error", detail="OPENAI_API_KEY missing")
    else:
        llm_status = SubsystemStatus(status="ok", detail=settings.llm_provider)

    overall = "ok" if vector_db.status == "ok" and llm_status.status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        api="ok",
        vector_db=vector_db,
        llm=llm_status,
        vector_count=vector_count,
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query(req: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    svc = get_rag_service()
    logger.info("query_received", request_id=request_id, query=req.query, namespace=req.namespace)
    start = time.perf_counter()
    try:
        with QUERY_LATENCY_SECONDS.time():
            result = svc.answer(
                query=req.query,
                namespace=req.namespace,
                vector_store=req.vector_store,
                metadata_filter=req.filters,
                k=req.k,
            )
    except LLMGenerationError as exc:
        QUERIES_TOTAL.labels(outcome="error").inc()
        logger.error("query_failed", request_id=request_id, error=repr(exc))
        raise HTTPException(status_code=503, detail={"error": "llm_unavailable", "message": str(exc)})

    latency_ms = result.get("latency_ms", int((time.perf_counter() - start) * 1000))
    outcome = "answered" if result["sources"] else "no_results"
    QUERIES_TOTAL.labels(outcome=outcome).inc()

    response = QueryResponse(
        request_id=request_id,
        query=req.query,
        answer=result["answer"],
        sources=result["sources"],
        confidence_score=result["confidence_score"],
        latency_ms=latency_ms,
        model_used=result["model_used"],
        timestamp=datetime.now(timezone.utc),
    )

    # Audit log (FR-N8N-002 step 6).
    get_metadata_db().log_query(
        request_id=request_id,
        query=req.query,
        namespace=req.namespace,
        answer=result["answer"],
        confidence_score=result["confidence_score"],
        latency_ms=latency_ms,
        sources=[s["chunk_id"] for s in result["sources"]],
    )
    logger.info(
        "query_answered",
        request_id=request_id,
        confidence=result["confidence_score"],
        latency_ms=latency_ms,
        sources=len(result["sources"]),
    )
    return response


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(req: IngestRequest) -> IngestResponse:
    request_id = str(uuid.uuid4())
    pipeline = IngestionPipeline()
    logger.info("ingest_received", request_id=request_id, namespace=req.namespace)

    tmp_path: str | None = None
    try:
        if req.directory:
            path, is_dir = req.directory, True
        elif req.file_path:
            path, is_dir = req.file_path, False
        elif req.content_base64 and req.filename:
            suffix = os.path.splitext(req.filename)[1] or ".txt"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(base64.b64decode(req.content_base64))
            path, is_dir = tmp_path, False
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide one of: file_path, directory, or content_base64+filename.",
            )

        event = pipeline.run(
            path=path,
            is_directory=is_dir,
            namespace=req.namespace,
            vector_store=req.vector_store,
            force=req.force,
            request_id=request_id,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return IngestResponse(
        request_id=event["request_id"],
        status=event["status"],
        doc_name=os.path.basename(event["doc_name"]),
        namespace=event["namespace"],
        chunks_total=event["chunks_total"],
        chunks_upserted=event["chunks_upserted"],
        chunks_skipped=event["chunks_skipped"],
        duration_ms=event["duration_ms"],
        errors=event["errors"],
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)

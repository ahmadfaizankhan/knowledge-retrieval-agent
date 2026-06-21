"""Pydantic request/response models (FR-GEN-002)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    similarity_score: float
    excerpt: str = Field(..., description="First 200 chars of the chunk text")


class QueryRequest(BaseModel):
    query: str
    namespace: str = "default"
    vector_store: Optional[str] = Field(
        default=None, description="Override: 'chroma' or 'pinecone'"
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None, description="Metadata filter, e.g. {'doc_type': {'$in': ['policy']}}"
    )
    k: Optional[int] = None


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    sources: list[Source]
    confidence_score: float
    latency_ms: int
    model_used: str
    timestamp: datetime


class IngestRequest(BaseModel):
    file_path: Optional[str] = Field(default=None, description="Local path to a document")
    directory: Optional[str] = Field(default=None, description="Directory to bulk-ingest")
    content_base64: Optional[str] = Field(
        default=None, description="Base64-encoded document content"
    )
    filename: Optional[str] = Field(
        default=None, description="Filename (required with content_base64)"
    )
    namespace: str = "default"
    vector_store: Optional[str] = None
    force: bool = False


class IngestResponse(BaseModel):
    request_id: str
    status: str
    doc_name: str
    namespace: str
    chunks_total: int
    chunks_upserted: int
    chunks_skipped: int
    duration_ms: int
    errors: list[dict[str, Any]] = Field(default_factory=list)


class SubsystemStatus(BaseModel):
    status: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    api: str
    vector_db: SubsystemStatus
    llm: SubsystemStatus
    vector_count: int
    timestamp: datetime

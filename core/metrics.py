"""Prometheus metrics (FR-REL-004).

Exposes the four metric families required by the PRD plus a couple of helpers.
All metrics live in a dedicated registry so the FastAPI ``/metrics`` endpoint
can render them deterministically.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

QUERY_LATENCY_SECONDS = Histogram(
    "query_latency_seconds",
    "End-to-end query latency in seconds.",
    buckets=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)

INGESTION_CHUNKS_TOTAL = Counter(
    "ingestion_chunks_total",
    "Total number of chunks upserted across all ingestion runs.",
    registry=REGISTRY,
)

RETRIEVAL_SCORE_MEAN = Gauge(
    "retrieval_score_mean",
    "Mean similarity score of the most recent retrieval.",
    registry=REGISTRY,
)

LLM_TOKENS_USED_TOTAL = Counter(
    "llm_tokens_used_total",
    "Total number of LLM tokens consumed (prompt + completion).",
    registry=REGISTRY,
)

QUERIES_TOTAL = Counter(
    "queries_total",
    "Total number of queries served, labelled by outcome.",
    labelnames=("outcome",),
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Render the registry in Prometheus text exposition format."""
    return generate_latest(REGISTRY)

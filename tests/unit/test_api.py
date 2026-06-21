"""API tests using FastAPI's TestClient (offline config)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api"] == "ok"
    assert body["llm"]["status"] == "ok"


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"query_latency_seconds" in resp.content
    assert b"ingestion_chunks_total" in resp.content


def test_query_returns_sources(seeded_namespace):
    resp = client.post(
        "/query",
        json={"query": "What is the refund window?", "namespace": seeded_namespace},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["sources"]
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["latency_ms"] >= 0
    assert "request_id" in body


def test_query_no_results_returns_fallback(seeded_namespace):
    resp = client.post(
        "/query",
        json={"query": "zxqwv nonsense gibberish token", "namespace": seeded_namespace},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Either fallback message or empty sources, but never an error.
    assert isinstance(body["sources"], list)


def test_ingest_via_path(sample_docs_dir):
    resp = client.post(
        "/ingest",
        json={
            "file_path": f"{sample_docs_dir}/refund_policy.md",
            "namespace": "test",
            "force": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_upserted"] > 0


def test_api_key_enforced(monkeypatch):
    """When REQUIRE_API_KEY is on, missing header -> 401."""
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("REQUIRE_API_KEY", "true")
    monkeypatch.setenv("FASTAPI_API_KEY", "secret")
    try:
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 401
        ok = client.post(
            "/query",
            json={"query": "test", "namespace": "test"},
            headers={"X-API-Key": "secret"},
        )
        assert ok.status_code == 200
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()

"""Shared FastAPI dependencies: API-key auth and singleton services."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException, status

from config.settings import get_settings
from retrieval.chain import RAGService


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Process-wide singleton RAG service (stateless, FR-SCA-003)."""
    return RAGService()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Validate the ``X-API-Key`` header (FR-SEC-005).

    Enforcement is controlled by ``REQUIRE_API_KEY``. When enforced and the
    configured key is missing/incorrect, returns HTTP 401.
    """
    settings = get_settings()
    if not settings.require_api_key:
        return
    expected = settings.fastapi_api_key
    if not expected or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )

"""Optional cross-encoder reranking (FR-RET-003, stretch goal).

Guarded by ``ENABLE_RERANKER``. Falls back to a no-op identity reranker if
``sentence-transformers`` is not installed so the rest of the system keeps
working offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import Settings, get_settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.documents import Document

logger = get_logger("retrieval.reranker")


class CrossEncoderReranker:
    """Re-orders candidate chunks by cross-encoder relevance to the query."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.reranker_model)
            logger.info("reranker_loaded", model=self.settings.reranker_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker_unavailable", error=repr(exc))

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(
        self, query: str, chunks: list["Document"], top_n: int | None = None
    ) -> list["Document"]:
        if not chunks:
            return chunks
        top_n = top_n or len(chunks)
        if self._model is None:
            # Identity fallback: preserve original order.
            return chunks[:top_n]
        pairs = [(query, c.page_content) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        for chunk, score in ranked:
            chunk.metadata["rerank_score"] = float(score)
        return [c for c, _ in ranked[:top_n]]

"""Pluggable embedding backend factory (FR-VEC-003).

Selects an embedding implementation based on ``EMBEDDING_PROVIDER``:

* ``openai``      -> ``OpenAIEmbeddings`` (text-embedding-3-large/small)
* ``huggingface`` -> ``HuggingFaceEmbeddings`` (BAAI/bge-large-en-v1.5, ...)
* ``local``       -> deterministic offline embeddings (no API key required)

OpenAI calls are wrapped with exponential-backoff retry via ``tenacity``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Settings, get_settings
from core.logging import get_logger
from embeddings.local_embeddings import LOCAL_EMBEDDING_DIM, LocalDeterministicEmbeddings

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.embeddings import Embeddings

logger = get_logger("embeddings.embedder")


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Return OpenAI transient error types if the SDK is importable."""
    try:
        import openai

        return (openai.RateLimitError, openai.APIError, openai.APITimeoutError)
    except Exception:  # noqa: BLE001
        return (Exception,)


class RetryingOpenAIEmbeddings:
    """Decorator that adds tenacity retry around OpenAIEmbeddings calls."""

    def __init__(self, inner: "Embeddings"):
        self._inner = inner
        exc_types = _retryable_exceptions()
        self._retry = retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(exc_types),
        )

    def embed_documents(self, texts):
        return self._retry(self._inner.embed_documents)(texts)

    def embed_query(self, text):
        return self._retry(self._inner.embed_query)(text)

    def __getattr__(self, item):  # delegate everything else
        return getattr(self._inner, item)


class EmbeddingFactory:
    """Creates the configured embedding backend."""

    @staticmethod
    def create(settings: Settings | None = None) -> "Embeddings":
        settings = settings or get_settings()
        provider = settings.embedding_provider

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            settings.require_openai()
            base = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
            logger.info("embedding_backend", provider="openai", model=settings.embedding_model)
            return RetryingOpenAIEmbeddings(base)  # type: ignore[return-value]

        if provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(
                "embedding_backend", provider="huggingface", model=settings.embedding_model
            )
            return HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # Default: local deterministic embeddings.
        logger.info("embedding_backend", provider="local", dimension=LOCAL_EMBEDDING_DIM)
        return LocalDeterministicEmbeddings(dimension=LOCAL_EMBEDDING_DIM)

    @staticmethod
    def dimension(settings: Settings | None = None) -> int:
        """Return the output dimension for the configured backend."""
        settings = settings or get_settings()
        if settings.embedding_provider == "local":
            return LOCAL_EMBEDDING_DIM
        return settings.embedding_dimension

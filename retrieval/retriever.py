"""Retriever construction (FR-RET-001/002).

Builds MMR or similarity retrievers over a LangChain vector store, with support
for metadata filter injection.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from config.settings import Settings, get_settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore

logger = get_logger("retrieval.retriever")


class RetrieverFactory:
    """Creates configured retrievers from a vector store."""

    @staticmethod
    def create_mmr_retriever(
        vector_store: "VectorStore",
        k: int,
        fetch_k: int,
        lambda_mult: float,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> "BaseRetriever":
        search_kwargs: dict[str, Any] = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    @staticmethod
    def create_similarity_retriever(
        vector_store: "VectorStore",
        k: int,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> "BaseRetriever":
        search_kwargs: dict[str, Any] = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
            return vector_store.as_retriever(
                search_type="similarity_score_threshold", search_kwargs=search_kwargs
            )
        return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    @classmethod
    def create(
        cls,
        vector_store: "VectorStore",
        settings: Settings | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> "BaseRetriever":
        """Create the retriever configured by ``RETRIEVAL_STRATEGY``."""
        settings = settings or get_settings()
        if settings.retrieval_strategy == "similarity":
            return cls.create_similarity_retriever(
                vector_store,
                k=settings.retrieval_k,
                score_threshold=settings.effective_score_threshold,
                metadata_filter=metadata_filter,
            )
        return cls.create_mmr_retriever(
            vector_store,
            k=settings.retrieval_k,
            fetch_k=settings.retrieval_fetch_k,
            lambda_mult=settings.retrieval_lambda_mult,
            score_threshold=settings.effective_score_threshold,
            metadata_filter=metadata_filter,
        )

"""Custom exception hierarchy for the knowledge retrieval agent."""

from __future__ import annotations


class KnowledgeAgentError(Exception):
    """Base class for all application-specific errors."""


class DocumentLoadError(KnowledgeAgentError):
    """Raised when a document cannot be loaded by any loader."""

    def __init__(self, file_path: str, original: Exception | None = None):
        self.file_path = file_path
        self.original = original
        msg = f"Failed to load document '{file_path}'"
        if original is not None:
            msg += f": {original!r}"
        super().__init__(msg)


class PineconeConfigError(KnowledgeAgentError):
    """Raised when an existing Pinecone index has an incompatible config."""


class EmbeddingError(KnowledgeAgentError):
    """Raised when embedding generation fails after retries."""


class VectorStoreError(KnowledgeAgentError):
    """Raised on vector store connection or upsert failures."""


class LLMGenerationError(KnowledgeAgentError):
    """Raised when the LLM generation call fails after retries."""

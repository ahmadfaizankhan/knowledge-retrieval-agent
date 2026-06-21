"""Text chunking, metadata enrichment and SHA-256 dedup hashing (FR-ING-002/003)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from config.settings import get_settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

logger = get_logger("ingestion.splitter")


def compute_chunk_hash(text: str) -> str:
    """Return the SHA-256 hex digest of a chunk's text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkingEngine:
    """Splits documents into chunks and enriches their metadata."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    # --- Splitting strategies ----------------------------------------------
    def split_recursive(
        self,
        docs: list["Document"],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list["Document"]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = chunk_size or self.settings.chunk_size
        chunk_overlap = chunk_overlap or self.settings.chunk_overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)
        return self._enrich(chunks)

    def split_semantic(
        self, docs: list["Document"], embeddings: "Embeddings"
    ) -> list["Document"]:
        from langchain_experimental.text_splitter import SemanticChunker

        splitter = SemanticChunker(embeddings)
        chunks = splitter.split_documents(docs)
        return self._enrich(chunks)

    def split(
        self, docs: list["Document"], embeddings: "Embeddings | None" = None
    ) -> list["Document"]:
        """Dispatch to the configured chunking strategy."""
        if self.settings.chunking_strategy == "semantic":
            if embeddings is None:
                raise ValueError("Semantic chunking requires an embeddings instance.")
            return self.split_semantic(docs, embeddings)
        return self.split_recursive(docs)

    # --- Metadata enrichment + filtering -----------------------------------
    def _enrich(self, chunks: list["Document"]) -> list["Document"]:
        """Inject chunk_index, timestamp, hash and a stable chunk_id."""
        timestamp = datetime.now(timezone.utc).isoformat()
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["ingestion_timestamp"] = timestamp
            chunk.metadata["chunk_hash"] = compute_chunk_hash(chunk.page_content)
            # UUID-based id avoids collisions across concurrent workers (FR-SCA-001).
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            # Normalise page metadata key produced by various loaders.
            if "page" in chunk.metadata and "page_number" not in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"]
        return chunks

    def filter_short_chunks(
        self, chunks: list["Document"], min_words: int | None = None
    ) -> list["Document"]:
        """Discard noise chunks below the minimum word count."""
        min_words = min_words if min_words is not None else self.settings.min_chunk_words
        kept = [c for c in chunks if len(c.page_content.split()) >= min_words]
        discarded = len(chunks) - len(kept)
        if discarded:
            logger.info("filtered_short_chunks", discarded=discarded, kept=len(kept))
        return kept

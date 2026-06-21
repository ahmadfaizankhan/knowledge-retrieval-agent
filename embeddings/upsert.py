"""Vector-store management: Chroma (dev) and Pinecone (prod) (FR-VEC-001/002).

The :class:`VectorStoreManager` handles deduplication, batched upserts and
structured logging, and returns a LangChain vector store object for retrieval.
Pinecone imports are lazy so the dev path never requires the Pinecone SDK.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from config.settings import Settings, get_settings
from core.exceptions import PineconeConfigError, VectorStoreError
from core.logging import get_logger
from core.metadata_db import MetadataDB, get_metadata_db
from core.metrics import INGESTION_CHUNKS_TOTAL
from embeddings.embedder import EmbeddingFactory

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

logger = get_logger("embeddings.upsert")


class VectorStoreManager:
    """Creates/connects vector stores and performs deduplicated upserts."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: "Embeddings | None" = None,
        metadata_db: MetadataDB | None = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embeddings or EmbeddingFactory.create(self.settings)
        self.metadata_db = metadata_db or get_metadata_db()
        # Cache store handles so we don't open a new client per call (avoids
        # SQLite/Chroma contention under concurrent load).
        self._store_cache: dict[tuple[str, str], Any] = {}
        self._store_lock = threading.Lock()

    # --- Dedup -------------------------------------------------------------
    def check_duplicate(self, chunk_hash: str) -> bool:
        """Return True if this chunk hash has already been upserted."""
        return self.metadata_db.hash_exists(chunk_hash)

    def _dedupe(
        self, chunks: list["Document"], namespace: str, force: bool
    ) -> tuple[list["Document"], int]:
        if force:
            return chunks, 0
        unique: list["Document"] = []
        skipped = 0
        for chunk in chunks:
            h = chunk.metadata.get("chunk_hash")
            if h and self.check_duplicate(h):
                skipped += 1
                continue
            unique.append(chunk)
        return unique, skipped

    def _record_hashes(self, chunks: list["Document"], namespace: str) -> None:
        for c in chunks:
            self.metadata_db.record_hash(
                chunk_hash=c.metadata.get("chunk_hash", ""),
                chunk_id=c.metadata.get("chunk_id", ""),
                source_file=c.metadata.get("source_file"),
                namespace=namespace,
            )

    # --- Chroma ------------------------------------------------------------
    def get_chroma_store(self, collection_name: str | None = None):
        from langchain_chroma import Chroma

        collection_name = collection_name or self.settings.chroma_collection
        cache_key = ("chroma", collection_name)
        with self._store_lock:
            if cache_key in self._store_cache:
                return self._store_cache[cache_key]
            store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir,
                # Cosine space keeps relevance scores in a meaningful [0, 1] range.
                collection_metadata={"hnsw:space": "cosine"},
            )
            self._store_cache[cache_key] = store
            return store

    def upsert_chroma(
        self,
        chunks: list["Document"],
        collection_name: str | None = None,
        namespace: str = "default",
        force: bool = False,
    ) -> int:
        start = time.perf_counter()
        collection_name = collection_name or self.settings.chroma_collection
        unique, skipped = self._dedupe(chunks, namespace, force)
        upserted = 0
        if unique:
            store = self.get_chroma_store(collection_name)
            ids = [c.metadata["chunk_id"] for c in unique]
            store.add_documents(documents=unique, ids=ids)
            self._record_hashes(unique, namespace)
            upserted = len(unique)
        INGESTION_CHUNKS_TOTAL.inc(upserted)
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "upsert_chroma",
            collection=collection_name,
            chunks_total=len(chunks),
            chunks_upserted=upserted,
            chunks_skipped=skipped,
            duration_ms=duration_ms,
        )
        return upserted

    # --- Pinecone ----------------------------------------------------------
    def _pinecone_client(self):
        from pinecone import Pinecone

        return Pinecone(api_key=self.settings.require_pinecone())

    def ensure_pinecone_index(self, create_if_missing: bool = True):
        """Verify (and optionally create) the Pinecone index (FR-VEC-002)."""
        from pinecone import ServerlessSpec

        pc = self._pinecone_client()
        index_name = self.settings.pinecone_index_name
        dimension = EmbeddingFactory.dimension(self.settings)
        existing = {i["name"] for i in pc.list_indexes()}
        if index_name not in existing:
            if not create_if_missing:
                raise PineconeConfigError(f"Pinecone index '{index_name}' does not exist.")
            logger.info("pinecone_create_index", index=index_name, dimension=dimension)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
        else:
            desc = pc.describe_index(index_name)
            if desc.dimension != dimension:
                raise PineconeConfigError(
                    f"Index '{index_name}' dimension {desc.dimension} != "
                    f"expected {dimension}."
                )
            if desc.metric != "cosine":
                raise PineconeConfigError(
                    f"Index '{index_name}' metric {desc.metric} != cosine."
                )
        return pc

    def get_pinecone_store(self, namespace: str | None = None):
        from langchain_pinecone import PineconeVectorStore

        namespace = namespace or self.settings.pinecone_namespace
        cache_key = ("pinecone", namespace)
        with self._store_lock:
            if cache_key in self._store_cache:
                return self._store_cache[cache_key]
            self.ensure_pinecone_index(create_if_missing=True)
            store = PineconeVectorStore(
                index_name=self.settings.pinecone_index_name,
                embedding=self.embeddings,
                namespace=namespace,
                pinecone_api_key=self.settings.require_pinecone(),
            )
            self._store_cache[cache_key] = store
            return store

    def upsert_pinecone(
        self,
        chunks: list["Document"],
        namespace: str | None = None,
        force: bool = False,
    ) -> int:
        start = time.perf_counter()
        namespace = namespace or self.settings.pinecone_namespace
        unique, skipped = self._dedupe(chunks, namespace, force)
        upserted = 0
        if unique:
            store = self.get_pinecone_store(namespace)
            batch_size = self.settings.pinecone_upsert_batch_size
            for i in range(0, len(unique), batch_size):
                batch = unique[i : i + batch_size]
                ids = [c.metadata["chunk_id"] for c in batch]
                store.add_documents(documents=batch, ids=ids, namespace=namespace)
            self._record_hashes(unique, namespace)
            upserted = len(unique)
        INGESTION_CHUNKS_TOTAL.inc(upserted)
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "upsert_pinecone",
            namespace=namespace,
            chunks_total=len(chunks),
            chunks_upserted=upserted,
            chunks_skipped=skipped,
            duration_ms=duration_ms,
        )
        return upserted

    # --- Unified API -------------------------------------------------------
    def upsert(
        self,
        chunks: list["Document"],
        vector_store: str | None = None,
        namespace: str = "default",
        force: bool = False,
    ) -> dict[str, int]:
        vector_store = vector_store or self.settings.vector_store
        _, skipped = self._dedupe(chunks, namespace, force) if not force else (chunks, 0)
        if vector_store == "pinecone":
            upserted = self.upsert_pinecone(chunks, namespace=namespace, force=force)
        elif vector_store == "chroma":
            upserted = self.upsert_chroma(
                chunks, namespace=namespace, force=force
            )
        else:
            raise VectorStoreError(f"Unknown vector store: {vector_store}")
        return {
            "chunks_total": len(chunks),
            "chunks_upserted": upserted,
            "chunks_skipped": len(chunks) - upserted,
        }

    def get_store(self, vector_store: str | None = None, namespace: str = "default"):
        """Return a vector store object suitable for retrieval."""
        vector_store = vector_store or self.settings.vector_store
        if vector_store == "pinecone":
            return self.get_pinecone_store(namespace)
        return self.get_chroma_store()

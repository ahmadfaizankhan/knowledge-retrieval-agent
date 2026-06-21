"""Multi-format document loading (FR-ING-001).

A :class:`DocumentLoaderFactory` routes each path to the correct LangChain
loader based on its file extension and supports bulk directory ingestion.
Loader dependencies are imported lazily so the module imports cleanly even when
an optional loader backend (e.g. ``unstructured``) is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import DocumentLoadError
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.documents import Document

logger = get_logger("ingestion.loader")

# Extensions handled by first-class loaders.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
# Glob used by directory ingestion.
DIRECTORY_GLOB = "**/*"


class DocumentLoaderFactory:
    """Factory that returns LangChain ``Document`` objects for any path."""

    @staticmethod
    def _pdf_loader(path: str):
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(path)

    @staticmethod
    def _text_loader(path: str):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(path, encoding="utf-8", autodetect_encoding=True)

    @staticmethod
    def _docx_loader(path: str):
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(path)

    @staticmethod
    def _unstructured_loader(path: str):
        from langchain_community.document_loaders import UnstructuredFileLoader

        return UnstructuredFileLoader(path)

    @classmethod
    def _loader_for(cls, file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return cls._pdf_loader(file_path)
        if ext in {".txt", ".md"}:
            return cls._text_loader(file_path)
        if ext == ".docx":
            return cls._docx_loader(file_path)
        # Fallback for unknown types.
        return cls._unstructured_loader(file_path)

    @classmethod
    def load(cls, file_path: str) -> list["Document"]:
        """Load a single file into a list of ``Document`` objects.

        Raises :class:`DocumentLoadError` on any failure.
        """
        if not os.path.exists(file_path):
            raise DocumentLoadError(file_path, FileNotFoundError(file_path))

        ext = Path(file_path).suffix.lower()
        doc_type = ext.lstrip(".") or "unknown"
        try:
            loader = cls._loader_for(file_path)
            docs = loader.load()
        except DocumentLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap any backend error
            logger.error("document_load_failed", file=file_path, error=repr(exc))
            raise DocumentLoadError(file_path, exc) from exc

        # Stamp consistent base metadata on every document.
        for doc in docs:
            doc.metadata.setdefault("source_file", os.path.basename(file_path))
            doc.metadata.setdefault("source_path", os.path.abspath(file_path))
            doc.metadata["doc_type"] = doc_type
        logger.info("document_loaded", file=file_path, doc_type=doc_type, pages=len(docs))
        return docs

    @classmethod
    def load_directory(
        cls, dir_path: str, glob: str = DIRECTORY_GLOB
    ) -> list["Document"]:
        """Bulk-load every supported file under ``dir_path``.

        Uses per-file loading (rather than ``DirectoryLoader`` directly) so a
        single bad file does not abort the whole batch (fail-forward).
        """
        if not os.path.isdir(dir_path):
            raise DocumentLoadError(dir_path, NotADirectoryError(dir_path))

        all_docs: list["Document"] = []
        files = sorted(
            p
            for p in Path(dir_path).rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("directory_scan", dir=dir_path, files_found=len(files))
        for path in files:
            try:
                all_docs.extend(cls.load(str(path)))
            except DocumentLoadError as exc:
                # Fail-forward: log and continue with the rest of the batch.
                logger.error("skip_unloadable_file", file=str(path), error=repr(exc))
        return all_docs

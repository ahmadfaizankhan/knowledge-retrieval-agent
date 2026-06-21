"""Unit tests for the document loader factory."""

from __future__ import annotations

import pytest

from core.exceptions import DocumentLoadError
from ingestion.loader import DocumentLoaderFactory


def test_load_markdown(sample_docs_dir):
    docs = DocumentLoaderFactory.load(f"{sample_docs_dir}/refund_policy.md")
    assert docs
    assert all(d.metadata["doc_type"] == "md" for d in docs)
    assert any("refund" in d.page_content.lower() for d in docs)


def test_load_txt(sample_docs_dir):
    docs = DocumentLoaderFactory.load(f"{sample_docs_dir}/employee_handbook.txt")
    assert docs
    assert docs[0].metadata["source_file"] == "employee_handbook.txt"
    assert docs[0].metadata["doc_type"] == "txt"


def test_load_pdf(sample_docs_dir):
    docs = DocumentLoaderFactory.load(f"{sample_docs_dir}/data_retention_policy.pdf")
    assert docs
    assert docs[0].metadata["doc_type"] == "pdf"
    assert "retention" in " ".join(d.page_content.lower() for d in docs)


def test_load_docx(sample_docs_dir):
    docs = DocumentLoaderFactory.load(f"{sample_docs_dir}/it_security_policy.docx")
    assert docs
    assert docs[0].metadata["doc_type"] == "docx"
    assert "encryption" in " ".join(d.page_content.lower() for d in docs)


def test_load_missing_file_raises():
    with pytest.raises(DocumentLoadError):
        DocumentLoaderFactory.load("does/not/exist.pdf")


def test_load_directory(sample_docs_dir):
    docs = DocumentLoaderFactory.load_directory(sample_docs_dir)
    sources = {d.metadata["source_file"] for d in docs}
    assert "refund_policy.md" in sources
    assert "financial_report.md" in sources
    assert "employee_handbook.txt" in sources


def test_load_directory_bad_path_raises():
    with pytest.raises(DocumentLoadError):
        DocumentLoaderFactory.load_directory("no/such/dir")

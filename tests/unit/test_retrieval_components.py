"""Unit tests for retriever, reranker and vector-store manager helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from embeddings.upsert import VectorStoreManager
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import RetrieverFactory


def test_mmr_retriever_built(settings, seeded_namespace):
    vsm = VectorStoreManager(settings)
    store = vsm.get_chroma_store()
    retriever = RetrieverFactory.create_mmr_retriever(
        store, k=3, fetch_k=10, lambda_mult=0.5
    )
    docs = retriever.invoke("refund window")
    assert isinstance(docs, list)


def test_similarity_retriever_built(settings, seeded_namespace):
    vsm = VectorStoreManager(settings)
    store = vsm.get_chroma_store()
    retriever = RetrieverFactory.create_similarity_retriever(store, k=3)
    docs = retriever.invoke("vacation days")
    assert isinstance(docs, list)


def test_retriever_factory_dispatch(settings, seeded_namespace):
    vsm = VectorStoreManager(settings)
    store = vsm.get_chroma_store()
    retriever = RetrieverFactory.create(store, settings)
    assert retriever is not None


def test_store_cache_returns_same_instance(settings):
    vsm = VectorStoreManager(settings)
    a = vsm.get_chroma_store()
    b = vsm.get_chroma_store()
    assert a is b


def test_reranker_identity_fallback_when_unavailable(settings):
    reranker = CrossEncoderReranker(settings)
    docs = [
        Document(page_content="alpha", metadata={"chunk_id": "1"}),
        Document(page_content="beta", metadata={"chunk_id": "2"}),
    ]
    out = reranker.rerank("alpha", docs, top_n=2)
    # Without sentence-transformers installed it preserves order; with it,
    # it still returns the requested number of docs.
    assert len(out) == 2


def test_reranker_handles_empty(settings):
    reranker = CrossEncoderReranker(settings)
    assert reranker.rerank("q", [], top_n=5) == []

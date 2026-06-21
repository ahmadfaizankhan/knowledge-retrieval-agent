"""Unit tests for the embedding factory and local embeddings."""

from __future__ import annotations

import math

from embeddings.embedder import EmbeddingFactory
from embeddings.local_embeddings import LOCAL_EMBEDDING_DIM, LocalDeterministicEmbeddings


def test_local_embedding_dimension(settings):
    emb = EmbeddingFactory.create(settings)
    vec = emb.embed_query("hello world")
    assert len(vec) == LOCAL_EMBEDDING_DIM
    assert EmbeddingFactory.dimension(settings) == LOCAL_EMBEDDING_DIM


def test_local_embedding_is_normalized():
    emb = LocalDeterministicEmbeddings()
    vec = emb.embed_query("the quick brown fox")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_local_embedding_deterministic():
    emb = LocalDeterministicEmbeddings()
    assert emb.embed_query("same text") == emb.embed_query("same text")


def test_similar_texts_closer_than_dissimilar():
    emb = LocalDeterministicEmbeddings()

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    base = emb.embed_query("refund policy thirty days")
    similar = emb.embed_query("refund policy within thirty days window")
    different = emb.embed_query("parental leave sixteen weeks")
    assert cos(base, similar) > cos(base, different)


def test_embed_documents_batch():
    emb = LocalDeterministicEmbeddings()
    vecs = emb.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3

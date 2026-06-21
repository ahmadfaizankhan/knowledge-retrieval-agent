"""Deterministic, dependency-free embeddings for offline dev / CI / tests.

Implements the LangChain ``Embeddings`` interface using a hashing trick: each
token is hashed into a fixed-dimension vector with a signed bucket, and the
resulting vector is L2-normalised. Texts that share vocabulary land close
together under cosine similarity, which is enough to exercise the full
ingest -> retrieve -> generate path without any external API.

This is NOT a semantic model — it is a stand-in so the system runs end-to-end
with no API keys. Set ``EMBEDDING_PROVIDER=openai`` (or ``huggingface``) for
real semantic embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Higher dimensionality keeps hash collisions (and therefore the similarity
# noise floor) low, so genuine lexical overlap stands out for short queries.
LOCAL_EMBEDDING_DIM = 2048


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LocalDeterministicEmbeddings(Embeddings):
    """Hashing-based deterministic embeddings."""

    def __init__(self, dimension: int = LOCAL_EMBEDDING_DIM):
        self.dimension = dimension

    @property
    def model_name(self) -> str:
        return f"local-hashing-{self.dimension}d"

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        # Presence-based (set) hashing: each distinct token contributes once, so
        # a frequently-repeated common word ("days") cannot dominate distinctive
        # vocabulary ("vacation"). This makes lexical retrieval far more robust
        # than raw term-frequency for the offline stand-in embedding.
        tokens = set(_tokenize(text))
        if not tokens:
            # Avoid zero vectors (undefined cosine); seed a constant feature.
            vec[0] = 1.0
            return vec
        for token in tokens:
            h = int.from_bytes(
                hashlib.md5(token.encode("utf-8")).digest()[:8], "little"
            )
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

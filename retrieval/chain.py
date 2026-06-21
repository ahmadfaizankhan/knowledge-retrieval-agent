"""RAG chain assembly and the high-level query service (FR-GEN-001/002/003).

``RAGChainBuilder`` exposes LangChain-native chains for PRD compliance.
``RAGService`` is the production query path used by the API: it retrieves with
scores, applies the score threshold + fallback guard, optionally reranks, and
returns a fully structured result (answer + per-source similarity scores).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from config.settings import Settings, get_settings
from core.exceptions import LLMGenerationError
from core.logging import get_logger
from core.metrics import LLM_TOKENS_USED_TOTAL, RETRIEVAL_SCORE_MEAN
from embeddings.upsert import VectorStoreManager
from retrieval.llm import LLMFactory
from retrieval.reranker import CrossEncoderReranker

logger = get_logger("retrieval.chain")

SYSTEM_PROMPT_TEMPLATE = """You are a precise knowledge retrieval assistant. Answer the user's question \
using ONLY the context provided below. If the answer cannot be found in the \
context, respond with: "I cannot find a verified answer to this question in \
the available documents."

Do NOT speculate, infer beyond the context, or use external knowledge.
Always cite the source document and chunk ID for every factual claim.

Context:
{context}

Question: {question}

Answer (with citations):"""

NO_RESULTS_RESPONSE = "No sufficiently relevant documents found."


class RAGChainBuilder:
    """Builds LangChain QA / conversational chains (PRD §2.5)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @property
    def prompt(self) -> PromptTemplate:
        return PromptTemplate(
            template=SYSTEM_PROMPT_TEMPLATE, input_variables=["context", "question"]
        )

    def build_qa_chain(self, retriever, llm=None):
        """Return a ``RetrievalQAWithSourcesChain`` (PRD-mandated base chain)."""
        from langchain.chains import RetrievalQAWithSourcesChain

        llm = llm or LLMFactory.create(self.settings)
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

    def build_conversational_chain(self, retriever, memory, llm=None):
        """Return a multi-turn ``ConversationalRetrievalChain``."""
        from langchain.chains import ConversationalRetrievalChain

        llm = llm or LLMFactory.create(self.settings)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
        )


def _dedupe_by_content(
    scored: list[tuple[Document, float]]
) -> list[tuple[Document, float]]:
    """Drop duplicate chunks (same text), keeping the highest-scoring copy."""
    best: dict[str, tuple[Document, float]] = {}
    order: list[str] = []
    for doc, score in scored:
        key = doc.page_content
        if key not in best:
            best[key] = (doc, score)
            order.append(key)
        elif score > best[key][1]:
            best[key] = (doc, score)
    return [best[k] for k in order]


def _format_context(scored: list[tuple[Document, float]]) -> str:
    blocks = []
    for doc, _score in scored:
        cid = doc.metadata.get("chunk_id", "unknown")
        src = doc.metadata.get("source_file", "unknown")
        blocks.append(f"[source_file={src} | chunk_id={cid}]\n{doc.page_content}")
    return "\n\n".join(blocks)


class RAGService:
    """End-to-end query service returning structured, citation-backed answers."""

    def __init__(
        self,
        settings: Settings | None = None,
        vsm: VectorStoreManager | None = None,
    ):
        self.settings = settings or get_settings()
        self.vsm = vsm or VectorStoreManager(self.settings)
        self.llm = LLMFactory.create(self.settings)
        self.prompt = PromptTemplate(
            template=SYSTEM_PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        self.reranker = (
            CrossEncoderReranker(self.settings) if self.settings.enable_reranker else None
        )

    def _retrieve_with_scores(
        self, query: str, namespace: str, vector_store: str, metadata_filter
    ) -> list[tuple[Document, float]]:
        store = self.vsm.get_store(vector_store=vector_store, namespace=namespace)
        # Over-fetch to give MMR/reranker room, then trim to k.
        fetch_k = max(self.settings.retrieval_fetch_k, self.settings.retrieval_k)
        kwargs: dict[str, Any] = {"k": fetch_k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        try:
            results = store.similarity_search_with_relevance_scores(query, **kwargs)
        except TypeError:
            # Some stores don't accept a filter kwarg signature; retry plainly.
            results = store.similarity_search_with_relevance_scores(query, k=fetch_k)
        return results

    def answer(
        self,
        query: str,
        namespace: str = "default",
        vector_store: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        vector_store = vector_store or self.settings.vector_store
        k = k or self.settings.retrieval_k

        scored = self._retrieve_with_scores(query, namespace, vector_store, metadata_filter)
        # Apply the score threshold (FR-RET-001), adjusted for the embedding backend.
        threshold = self.settings.effective_score_threshold
        passing = [(d, s) for d, s in scored if s >= threshold]
        # Collapse identical chunks (e.g. from repeated force re-seeding) so the
        # context — and the answer — is not duplicated.
        passing = _dedupe_by_content(passing)

        # Fallback guard: do NOT call the LLM with zero context (FR-GEN-003).
        if not passing:
            RETRIEVAL_SCORE_MEAN.set(0.0)
            return {
                "answer": NO_RESULTS_RESPONSE,
                "sources": [],
                "confidence_score": 0.0,
                "model_used": self._model_name(),
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }

        # Optional reranking, then trim to k.
        if self.reranker is not None and self.reranker.available:
            docs = [d for d, _ in passing]
            reranked = self.reranker.rerank(query, docs, top_n=k)
            score_by_id = {id(d): s for d, s in passing}
            passing = [(d, score_by_id.get(id(d), 0.0)) for d in reranked]
        passing = passing[:k]

        context = _format_context(passing)
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": query})
        except Exception as exc:  # noqa: BLE001
            logger.error("llm_generation_failed", error=repr(exc))
            raise LLMGenerationError(str(exc)) from exc

        # Rough token accounting for the metrics counter.
        LLM_TOKENS_USED_TOTAL.inc(_estimate_tokens(context) + _estimate_tokens(answer))

        scores = [s for _, s in passing]
        confidence = sum(scores) / len(scores) if scores else 0.0
        RETRIEVAL_SCORE_MEAN.set(confidence)

        sources = [self._to_source(d, s) for d, s in passing]
        return {
            "answer": answer.strip(),
            "sources": sources,
            "confidence_score": round(confidence, 4),
            "model_used": self._model_name(),
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }

    def _model_name(self) -> str:
        if self.settings.llm_provider == "openai":
            return self.settings.llm_model
        return "local-extractive"

    @staticmethod
    def _to_source(doc: Document, score: float) -> dict[str, Any]:
        return {
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page_number": doc.metadata.get("page_number"),
            "similarity_score": round(float(score), 4),
            "excerpt": doc.page_content[:200],
        }


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token heuristic; good enough for a usage counter.
    return max(1, len(text) // 4)

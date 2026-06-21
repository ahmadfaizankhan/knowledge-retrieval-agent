"""Integration tests for the retrieval + generation chain."""

from __future__ import annotations

import pytest

from retrieval.chain import NO_RESULTS_RESPONSE, RAGService


@pytest.fixture()
def rag_service(settings, vector_store):
    if vector_store == "pinecone" and not settings.pinecone_api_key:
        pytest.skip("Pinecone API key not configured; skipping pinecone integration tests.")
    return RAGService(settings)


def test_answer_refund_question(rag_service, seeded_namespace, vector_store):
    result = rag_service.answer(
        "What is the refund window in days?",
        namespace=seeded_namespace,
        vector_store=vector_store,
    )
    assert result["sources"]
    assert result["confidence_score"] > 0
    # The grounded answer should mention the 30-day window.
    assert "30" in result["answer"] or any(
        "30" in s["excerpt"] for s in result["sources"]
    )


def test_answer_cites_source_file(rag_service, seeded_namespace, vector_store):
    result = rag_service.answer(
        "How much annual revenue did the company report?",
        namespace=seeded_namespace,
        vector_store=vector_store,
    )
    files = {s["source_file"] for s in result["sources"]}
    assert "financial_report.md" in files


def test_no_results_fallback_skips_llm(rag_service, seeded_namespace, vector_store):
    result = rag_service.answer(
        "zzqx unrelated gibberish nonsense flarb",
        namespace=seeded_namespace,
        vector_store=vector_store,
    )
    if not result["sources"]:
        assert result["answer"] == NO_RESULTS_RESPONSE
        assert result["confidence_score"] == 0.0


def test_known_questions_have_citations(rag_service, seeded_namespace, vector_store):
    questions = [
        "How many vacation days do employees get?",
        "What is the parental leave duration?",
        "What was the net profit margin?",
    ]
    for q in questions:
        result = rag_service.answer(q, namespace=seeded_namespace, vector_store=vector_store)
        # Each answered query must carry at least one citation.
        if result["answer"] != NO_RESULTS_RESPONSE:
            assert result["sources"], f"No sources for: {q}"

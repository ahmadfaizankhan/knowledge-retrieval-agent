"""Unit tests for the local extractive LLM and fallback behaviour."""

from __future__ import annotations

from retrieval.chain import SYSTEM_PROMPT_TEMPLATE
from retrieval.llm import LocalExtractiveLLM


def _prompt(context: str, question: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(context=context, question=question)


def test_extractive_answer_uses_context():
    llm = LocalExtractiveLLM()
    context = (
        "The refund window is 30 days from purchase. "
        "Gift cards are non-refundable. "
        "Shipping costs are not refundable."
    )
    out = llm.invoke(_prompt(context, "What is the refund window?"))
    assert "30 days" in out


def test_refuses_without_relevant_context():
    llm = LocalExtractiveLLM()
    context = "The sky is blue. Water is wet."
    out = llm.invoke(_prompt(context, "What is the capital of France?"))
    assert "cannot find a verified answer" in out.lower()


def test_refuses_on_empty_context():
    llm = LocalExtractiveLLM()
    out = llm.invoke(_prompt("", "anything"))
    assert "cannot find a verified answer" in out.lower()

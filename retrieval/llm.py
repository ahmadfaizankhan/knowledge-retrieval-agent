"""LLM backend factory (FR-GEN-001).

* ``openai`` -> ``ChatOpenAI`` at temperature 0.0, max_tokens 1024.
* ``local``  -> a deterministic extractive answerer that uses ONLY the context
  embedded in the prompt (no network, for dev / CI / tests).

Both implement the LangChain ``LLM`` / chat interface so downstream chains are
provider-agnostic.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM

from config.settings import Settings, get_settings
from core.logging import get_logger

logger = get_logger("retrieval.llm")

_NO_ANSWER = (
    "I cannot find a verified answer to this question in the available documents."
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "what", "which", "who", "whom", "how", "when", "where", "why",
    "this", "that", "these", "those", "with", "as", "by", "at", "from", "be",
    "it", "its", "do", "does", "did", "can", "could", "should", "would", "will",
}


class LocalExtractiveLLM(LLM):
    """Deterministic extractive answerer over the prompt's Context section."""

    max_sentences: int = 4

    @property
    def _llm_type(self) -> str:
        return "local-extractive"

    def _parse_prompt(self, prompt: str) -> tuple[str, str]:
        """Split the PRD prompt template into (context, question)."""
        context, question = prompt, ""
        if "Context:" in prompt:
            context = prompt.split("Context:", 1)[1]
        if "Question:" in context:
            context, question = context.split("Question:", 1)
        if "Answer" in question:
            question = question.split("Answer", 1)[0]
        return context.strip(), question.strip()

    @staticmethod
    def _keywords(text: str) -> set[str]:
        return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        context, question = self._parse_prompt(prompt)
        if not context.strip():
            return _NO_ANSWER
        # Drop the bracketed citation header lines ([source_file=... chunk_id=...])
        # injected by the context formatter so they never leak into the answer.
        context = "\n".join(
            line for line in context.splitlines() if not line.strip().startswith("[source_file=")
        )
        q_words = self._keywords(question)
        sentences = [s.strip() for s in _SENTENCE_RE.split(context) if s.strip()]
        if not sentences:
            return _NO_ANSWER

        scored: list[tuple[int, int, str]] = []
        for idx, sent in enumerate(sentences):
            overlap = len(self._keywords(sent) & q_words)
            if overlap > 0:
                scored.append((overlap, -idx, sent))
        if not scored:
            # No lexical overlap -> refuse, per the guardrail.
            return _NO_ANSWER
        scored.sort(reverse=True)
        top = [self._clean(s) for _, _, s in scored[: self.max_sentences]]
        return " ".join(t for t in top if t)

    @staticmethod
    def _clean(sentence: str) -> str:
        """Strip markdown header fragments glued to the start of a sentence."""
        lines = [ln for ln in sentence.splitlines() if not ln.lstrip().startswith("#")]
        return " ".join(ln.strip() for ln in lines if ln.strip()).strip()


class LLMFactory:
    """Creates the configured chat/LLM backend."""

    @staticmethod
    def create(settings: Settings | None = None):
        settings = settings or get_settings()
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            settings.require_openai()
            logger.info("llm_backend", provider="openai", model=settings.llm_model)
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.openai_api_key,
            )
        logger.info("llm_backend", provider="local")
        return LocalExtractiveLLM()

"""RAGAS evaluation harness (Phase 4.3).

Runs every question in the eval dataset through the live RAG chain and computes
Faithfulness, Answer Relevancy, Context Recall and Context Precision, plus a
derived hallucination rate (1 - faithfulness).

If the ``ragas`` package + an OpenAI key are available, the official RAGAS
metrics are used. Otherwise a deterministic, dependency-free fallback evaluator
is used (embedding-cosine + lexical overlap) so the harness still produces a
report offline / in CI.

    python tests/eval/evaluate_hallucination.py \
        --dataset tests/eval/rag_eval_dataset.json \
        --output  tests/eval/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a standalone script: ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import get_settings
from core.logging import get_logger
from embeddings.embedder import EmbeddingFactory
from retrieval.chain import NO_RESULTS_RESPONSE, RAGService

logger = get_logger("eval.hallucination")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _ragas_available() -> bool:
    settings = get_settings()
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("ragas") is not None
    except Exception:  # noqa: BLE001
        return False


def evaluate_fallback(dataset: list[dict], namespace: str) -> dict:
    """Deterministic offline evaluator.

    * answer_relevancy = cosine(answer, ground_truth)
    * faithfulness     = fraction of answer tokens supported by retrieved context
    * context_recall   = fraction of ground_truth tokens present in context
    * context_precision= fraction of retrieved chunks lexically overlapping GT
    """
    settings = get_settings()
    svc = RAGService(settings)
    emb = EmbeddingFactory.create(settings)

    per_q = []
    rel_scores, faith_scores, recall_scores, prec_scores = [], [], [], []

    threshold = settings.effective_score_threshold
    for item in dataset:
        q = item["question"]
        gt = item.get("ground_truth", "")
        result = svc.answer(q, namespace=namespace)
        answer = result["answer"]
        sources = result["sources"]
        # Score faithfulness/recall against the FULL retrieved chunk text
        # (not the 200-char display excerpt) for an accurate grounding measure.
        retrieved = [
            (d, s)
            for d, s in svc._retrieve_with_scores(q, namespace, settings.vector_store, None)
            if s >= threshold
        ]
        ctx_text = " ".join(d.page_content for d, _ in retrieved) or " ".join(
            s["excerpt"] for s in sources
        )

        # Answer relevancy via embedding cosine.
        if answer and answer != NO_RESULTS_RESPONSE and gt:
            relevancy = max(0.0, _cosine(emb.embed_query(answer), emb.embed_query(gt)))
        else:
            relevancy = 0.0

        ans_tok = _tokens(answer) - {"i", "the", "a", "an", "is", "are", "to", "of"}
        ctx_tok = _tokens(ctx_text)
        gt_tok = _tokens(gt)

        faithfulness = (
            len(ans_tok & ctx_tok) / len(ans_tok) if ans_tok else (1.0 if not sources else 0.0)
        )
        recall = len(gt_tok & ctx_tok) / len(gt_tok) if gt_tok else 0.0
        precision = 0.0
        if sources:
            relevant = sum(1 for s in sources if _tokens(s["excerpt"]) & gt_tok)
            precision = relevant / len(sources)

        rel_scores.append(relevancy)
        faith_scores.append(faithfulness)
        recall_scores.append(recall)
        prec_scores.append(precision)
        per_q.append(
            {
                "question": q,
                "answer": answer,
                "faithfulness": round(faithfulness, 3),
                "answer_relevancy": round(relevancy, 3),
                "context_recall": round(recall, 3),
                "context_precision": round(precision, 3),
                "num_sources": len(sources),
            }
        )

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    faithfulness = avg(faith_scores)
    return {
        "evaluator": "offline-fallback",
        "n_questions": len(dataset),
        "faithfulness": faithfulness,
        "answer_relevancy": avg(rel_scores),
        "context_recall": avg(recall_scores),
        "context_precision": avg(prec_scores),
        "hallucination_rate": round(1 - faithfulness, 4),
        "per_question": per_q,
    }


def evaluate_ragas(dataset: list[dict], namespace: str) -> dict:
    """Official RAGAS evaluation (requires OpenAI + ragas)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    settings = get_settings()
    svc = RAGService(settings)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in dataset:
        result = svc.answer(item["question"], namespace=namespace)
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s["excerpt"] for s in result["sources"]] or [""])
        rows["ground_truth"].append(item.get("ground_truth", ""))

    ds = Dataset.from_dict(rows)
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    df = scores.to_pandas()
    f = float(df["faithfulness"].mean())
    return {
        "evaluator": "ragas",
        "n_questions": len(dataset),
        "faithfulness": round(f, 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "context_recall": round(float(df["context_recall"].mean()), 4),
        "context_precision": round(float(df["context_precision"].mean()), 4),
        "hallucination_rate": round(1 - f, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation harness")
    parser.add_argument("--dataset", default="tests/eval/rag_eval_dataset.json")
    parser.add_argument("--output", default="tests/eval/eval_report.json")
    parser.add_argument("--namespace", default="test")
    parser.add_argument(
        "--seed", action="store_true", help="Seed the sample docs before evaluating."
    )
    args = parser.parse_args(argv)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    if args.seed:
        from ingestion.pipeline import IngestionPipeline

        IngestionPipeline().run(
            path="tests/fixtures/sample_docs",
            is_directory=True,
            namespace=args.namespace,
            vector_store="chroma",
            force=True,
        )

    if _ragas_available():
        logger.info("eval_mode", mode="ragas")
        report = evaluate_ragas(dataset, args.namespace)
    else:
        logger.info("eval_mode", mode="offline-fallback")
        report = evaluate_fallback(dataset, args.namespace)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

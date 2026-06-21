"""Retrieval latency / load benchmark and ingestion throughput (Phase 4.4).

Examples:
    python scripts/benchmark_retrieval.py --queries tests/eval/rag_eval_dataset.json --n 100 --concurrency 1
    python scripts/benchmark_retrieval.py --queries ... --n 200 --concurrency 10
    python scripts/benchmark_retrieval.py --mode ingest --dir tests/fixtures/bulk_docs/ --vector-store chroma
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.pipeline import IngestionPipeline
from retrieval.chain import RAGService


def _load_queries(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    queries = []
    for item in data:
        q = item.get("question") or item.get("query")
        if q:
            queries.append(q)
    return queries


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def bench_query(args) -> int:
    svc = RAGService()
    queries = _load_queries(args.queries)
    if not queries:
        print(json.dumps({"error": "no queries found"}))
        return 1
    # Build the workload of size n by cycling through the queries.
    workload = [queries[i % len(queries)] for i in range(args.n)]

    latencies: list[float] = []
    errors = 0

    def run_one(q: str) -> float | None:
        nonlocal errors
        t0 = time.perf_counter()
        try:
            svc.answer(q, namespace=args.namespace)
            return (time.perf_counter() - t0) * 1000.0
        except Exception:  # noqa: BLE001
            errors += 1
            return None

    start = time.perf_counter()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for ms in ex.map(run_one, workload):
                if ms is not None:
                    latencies.append(ms)
    else:
        for q in workload:
            ms = run_one(q)
            if ms is not None:
                latencies.append(ms)
    wall = time.perf_counter() - start

    report = {
        "mode": "query",
        "n": args.n,
        "concurrency": args.concurrency,
        "errors": errors,
        "error_rate": round(errors / args.n, 4) if args.n else 0.0,
        "throughput_qps": round(args.n / wall, 2) if wall else 0.0,
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "p99": round(_percentile(latencies, 99), 2),
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def bench_ingest(args) -> int:
    pipeline = IngestionPipeline()
    t0 = time.perf_counter()
    event = pipeline.run(
        path=args.dir,
        is_directory=True,
        namespace=args.namespace,
        vector_store=args.vector_store,
        force=True,
    )
    elapsed = time.perf_counter() - t0
    chunks = event["chunks_total"]
    tokens = event["tokens_estimated"]
    report = {
        "mode": "ingest",
        "chunks_total": chunks,
        "duration_s": round(elapsed, 2),
        "chunks_per_second": round(chunks / elapsed, 2) if elapsed else 0.0,
        "tokens_per_second": round(tokens / elapsed, 2) if elapsed else 0.0,
    }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval / ingestion benchmark")
    parser.add_argument("--mode", choices=["query", "ingest"], default="query")
    parser.add_argument("--queries", help="Path to eval dataset JSON")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--namespace", default="test")
    parser.add_argument("--dir", help="Directory for ingest mode")
    parser.add_argument("--vector-store", choices=["chroma", "pinecone"], default="chroma")
    args = parser.parse_args(argv)

    if args.mode == "ingest":
        if not args.dir:
            parser.error("--dir is required in ingest mode")
        return bench_ingest(args)
    if not args.queries:
        parser.error("--queries is required in query mode")
    return bench_query(args)


if __name__ == "__main__":
    sys.exit(main())

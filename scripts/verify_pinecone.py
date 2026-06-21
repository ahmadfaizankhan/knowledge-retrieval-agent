"""Verify / create the Pinecone production index and print its stats (FR-VEC-002).

    python scripts/verify_pinecone.py --create-if-missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.logging import get_logger
from embeddings.embedder import EmbeddingFactory
from embeddings.upsert import VectorStoreManager

logger = get_logger("scripts.verify_pinecone")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Pinecone index health")
    parser.add_argument("--create-if-missing", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    vsm = VectorStoreManager(settings)
    pc = vsm.ensure_pinecone_index(create_if_missing=args.create_if_missing)

    index = pc.Index(settings.pinecone_index_name)
    stats = index.describe_index_stats()
    desc = pc.describe_index(settings.pinecone_index_name)

    report = {
        "index_name": settings.pinecone_index_name,
        "dimension": desc.dimension,
        "metric": desc.metric,
        "expected_dimension": EmbeddingFactory.dimension(settings),
        "total_vector_count": stats.get("total_vector_count", 0),
        "index_fullness": stats.get("index_fullness", 0.0),
        "namespaces": list(stats.get("namespaces", {}).keys()),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        logger.error("verify_pinecone_failed", error=repr(exc))
        print(json.dumps({"status": "error", "error": repr(exc)}))
        sys.exit(1)

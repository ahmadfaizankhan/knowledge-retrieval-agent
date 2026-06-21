"""Seed ChromaDB from a directory of documents (FR-VEC-001).

    python scripts/seed_chroma.py --dir ./docs/ --namespace dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.pipeline import IngestionPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed ChromaDB")
    parser.add_argument("--dir", required=True, help="Directory of documents")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    pipeline = IngestionPipeline()
    event = pipeline.run(
        path=args.dir,
        is_directory=True,
        namespace=args.namespace,
        vector_store="chroma",
        force=args.force,
    )
    print(json.dumps(event, indent=2))

    # Verify with collection.count().
    store = pipeline.vsm.get_chroma_store()
    print(f"Collection '{store._collection.name}' count: {store._collection.count()}")
    return 0 if event["status"] in ("complete", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())

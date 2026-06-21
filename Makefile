# Makefile for the Knowledge Retrieval Agent.
# On Windows, run targets from Git Bash / WSL, or invoke the commands directly.

PYTHON ?= python
VENV_PY := .venv/Scripts/python.exe
SAMPLE_DOCS := tests/fixtures/sample_docs

.PHONY: install seed-chroma verify-pinecone run-api test test-unit test-integration eval benchmark lint clean

install:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

seed-chroma:
	$(VENV_PY) scripts/seed_chroma.py --dir $(SAMPLE_DOCS) --namespace test --force

verify-pinecone:
	$(VENV_PY) scripts/verify_pinecone.py --create-if-missing

run-api:
	$(VENV_PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

test-unit:
	$(VENV_PY) -m pytest tests/unit/ -v --tb=short

test-integration:
	$(VENV_PY) -m pytest tests/integration/ -v --vector-store chroma

eval:
	$(VENV_PY) tests/eval/evaluate_hallucination.py --seed \
		--dataset tests/eval/rag_eval_dataset.json \
		--output tests/eval/eval_report.json

benchmark:
	$(VENV_PY) scripts/benchmark_retrieval.py \
		--queries tests/eval/rag_eval_dataset.json --n 100 --concurrency 1

test: lint test-unit test-integration eval

lint:
	$(VENV_PY) -m pyflakes api config core embeddings ingestion retrieval scripts || true

clean:
	rm -rf chroma_db logs __pycache__ .pytest_cache *.sqlite3 tests/eval/eval_report.json

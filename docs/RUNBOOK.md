# Operations Runbook — Knowledge Retrieval Agent

This runbook covers day-2 operations for the RAG pipeline.

---

## 1. Ingesting new documents

### Via CLI
```bash
# Single file
python -m ingestion.pipeline --file /path/to/report.pdf --namespace production --vector-store pinecone

# Whole directory
python -m ingestion.pipeline --dir /path/to/docs --namespace production --vector-store pinecone

# Force re-ingest (bypass SHA-256 dedup, e.g. after a document update)
python -m ingestion.pipeline --file /path/to/report.pdf --force
```

The pipeline prints a JSON completion event:
```json
{"status":"complete","doc_name":"...","chunks_total":N,"chunks_upserted":N,"chunks_skipped":N,"duration_ms":N,"errors":[]}
```

### Via API
```bash
curl -X POST http://localhost:8080/ingest \
  -H "X-API-Key: $FASTAPI_API_KEY" -H "Content-Type: application/json" \
  -d '{"file_path":"/path/to/report.pdf","namespace":"production","vector_store":"pinecone"}'
```
Base64 payloads are also accepted: `{"content_base64":"...","filename":"x.pdf"}`.

### Deduplication
Every chunk's SHA-256 hash is recorded in the metadata DB. Re-running ingestion
without `--force` skips already-ingested chunks (`chunks_skipped > 0`,
`chunks_upserted == 0`). Use `--force` to overwrite after a document changes.

---

## 2. Adding a new n8n workflow trigger

1. Open n8n at `http://localhost:5678`.
2. **Import** `n8n/workflows/doc_ingestion_workflow.json` or
   `n8n/workflows/query_routing_workflow.json`.
3. Set credentials: the HTTP Request nodes read `FASTAPI_API_KEY` from the n8n
   environment; the Postgres/Slack nodes need their own credentials configured.
4. To add a new trigger (e.g. a watch-folder or S3 event), insert the trigger
   node before **Switch: Document Classification** and wire its output into it.
5. **Activate** the workflow. The ingestion webhook is `POST /webhook/ingest`;
   the query webhook is `POST /webhook/ask`.
6. HTTP nodes are configured to retry 3× with a 2s delay; error branches log
   failures without aborting the run (≥30-day execution history retention).

---

## 3. Rotating API keys

1. Generate new keys in the provider console (OpenAI / Pinecone) or for the
   service (`FASTAPI_API_KEY`).
2. Update `.env` (never commit it). For containers, update the secret store /
   `env_file` and redeploy.
3. Update the `FASTAPI_API_KEY` value in the n8n environment so workflow HTTP
   nodes keep authenticating.
4. Restart the API: `docker compose up -d api` (or restart uvicorn).
5. Verify: `curl http://localhost:8080/health` and a test `/query`.
6. Revoke the old keys in the provider console.

> The app never logs key values. Confirm with:
> `git log --all -p | grep -E "sk-|PINECONE_API_KEY" | grep -v example` → expect no output.

---

## 4. Interpreting RAGAS evaluation output

Run:
```bash
python tests/eval/evaluate_hallucination.py --seed \
  --dataset tests/eval/rag_eval_dataset.json --output tests/eval/eval_report.json
```

`eval_report.json` fields:

| Metric | Meaning | PRD target (production) |
|--------|---------|--------------------------|
| `faithfulness` | Fraction of answer claims grounded in retrieved context | ≥ 0.90 |
| `answer_relevancy` | Semantic similarity of answer to the question | ≥ 0.85 |
| `context_recall` | Fraction of ground-truth covered by retrieved chunks | ≥ 0.80 |
| `context_precision` | Fraction of retrieved chunks that are relevant | ≥ 0.75 |
| `hallucination_rate` | `1 - faithfulness` | ≤ 0.05 |

- `"evaluator": "ragas"` → official RAGAS metrics (needs `LLM_PROVIDER=openai` + key + `ragas` installed).
- `"evaluator": "offline-fallback"` → deterministic lexical/embedding approximation (no keys). Use for plumbing checks; **do not** gate production releases on its `answer_relevancy`, which is limited by the local lexical embedding.

If a target fails on the RAGAS path, tune in this order (see `todo.md` Phase 2.4):
chunk size/overlap → score threshold → `k` → enable reranker.

---

## 5. Scaling the Pinecone index

- The index is created **serverless** (`cosine`, dimension matches the embedding
  model: 3072 for `text-embedding-3-large`, 1536 for `-small`). Serverless scales
  to ≥ 10M vectors with no manual capacity changes.
- Keep upserts batched (`PINECONE_UPSERT_BATCH_SIZE`, default 100) to respect
  rate limits.
- **Namespaces isolate collections / tenants** — never commingle tenants in one
  namespace. Pass `--namespace` on ingest and `namespace` in queries.
- Horizontal ingestion: run multiple workers concurrently; UUID `chunk_id`s
  prevent vector-ID collisions.
- The FastAPI service is stateless — scale it behind a load balancer.
- Health check index stats: `python scripts/verify_pinecone.py`.

---

## 6. Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/query` returns "No sufficiently relevant documents found." | Nothing above score threshold | Lower `SCORE_THRESHOLD`, ingest more docs, or check the namespace |
| `OPENAI_API_KEY is not set` | Missing key with `*_PROVIDER=openai` | Set the key or switch provider to `local` |
| 401 on `/query` | `REQUIRE_API_KEY=true` and bad/missing header | Send `X-API-Key: $FASTAPI_API_KEY` |
| `PineconeConfigError` | Index dimension/metric mismatch | Recreate the index or align `EMBEDDING_MODEL` |
| Duplicated sentences in answer | Repeated `--force` re-seeding | Harmless (deduped at query time); clean `chroma_db/` to reset |

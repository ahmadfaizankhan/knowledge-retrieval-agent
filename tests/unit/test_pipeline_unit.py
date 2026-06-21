"""Unit tests for the ingestion pipeline orchestration edge cases."""

from __future__ import annotations

from ingestion.pipeline import IngestionPipeline, main


def test_pipeline_load_error_returns_error_status(settings):
    pipeline = IngestionPipeline(settings)
    event = pipeline.run(
        path="no/such/file.pdf",
        is_directory=False,
        namespace="x",
        vector_store="chroma",
        force=True,
    )
    assert event["status"] == "error"
    assert event["errors"]
    assert event["chunks_upserted"] == 0


def test_pipeline_emits_completion_event(settings, sample_docs_dir):
    pipeline = IngestionPipeline(settings)
    event = pipeline.run(
        path=f"{sample_docs_dir}/financial_report.md",
        is_directory=False,
        namespace="unit",
        vector_store="chroma",
        force=True,
    )
    for key in (
        "status", "request_id", "doc_name", "chunks_total",
        "chunks_upserted", "chunks_skipped", "tokens_estimated",
        "duration_ms", "errors",
    ):
        assert key in event


def test_cli_main_runs(sample_docs_dir, capsys):
    rc = main(
        [
            "--file", f"{sample_docs_dir}/employee_handbook.txt",
            "--namespace", "cli", "--vector-store", "chroma", "--force",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"status"' in out

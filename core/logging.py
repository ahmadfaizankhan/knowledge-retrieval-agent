"""Structured logging configuration using ``structlog``.

Emits JSON log lines with the fields required by FR-REL-001:
``timestamp``, ``level``, ``component``, ``event`` and (where supplied)
``duration_ms`` and ``error``. Logs are written both to stdout and to a
rotating file under ``LOG_DIR``.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import structlog

from config.settings import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure structlog + stdlib logging once per process (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.jsonl", maxBytes=10 * 1024 * 1024, backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to a ``component`` name."""
    configure_logging()
    return structlog.get_logger().bind(component=component)


@contextmanager
def log_duration(
    logger: structlog.stdlib.BoundLogger, event: str, **fields: Any
) -> Iterator[dict[str, Any]]:
    """Context manager that logs ``event`` with elapsed ``duration_ms``.

    Yields a mutable dict so the caller can attach extra fields that should be
    emitted in the completion log line.
    """
    extra: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield extra
    except Exception as exc:  # noqa: BLE001 - we re-raise after logging
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.error(event, duration_ms=duration_ms, error=repr(exc), **fields, **extra)
        raise
    else:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(event, duration_ms=duration_ms, **fields, **extra)

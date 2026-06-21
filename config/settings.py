"""Centralised, validated application settings.

Loads configuration from environment variables (and a local ``.env`` file)
using ``pydantic-settings``. A single cached :class:`Settings` instance is
exposed via :func:`get_settings` and consumed everywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed view over the project's environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider keys ------------------------------------------------------
    openai_api_key: str | None = Field(default=None)
    pinecone_api_key: str | None = Field(default=None)

    # --- Pinecone -----------------------------------------------------------
    pinecone_index_name: str = "knowledge-retrieval-agent"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "default"
    pinecone_upsert_batch_size: int = 100

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: Literal["openai", "huggingface", "local"] = "local"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = 768
    chunk_overlap: int = 96
    min_chunk_words: int = 20
    chunking_strategy: Literal["recursive", "semantic"] = "recursive"

    # --- Retrieval ----------------------------------------------------------
    retrieval_k: int = 6
    retrieval_fetch_k: int = 20
    retrieval_lambda_mult: float = 0.5
    retrieval_strategy: Literal["mmr", "similarity"] = "mmr"
    score_threshold: float = 0.72
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM generation -----------------------------------------------------
    llm_provider: Literal["openai", "local"] = "local"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    low_confidence_threshold: float = 0.65

    # --- Vector store -------------------------------------------------------
    vector_store: Literal["chroma", "pinecone"] = "chroma"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "knowledge_base_dev"

    # --- API ----------------------------------------------------------------
    fastapi_api_key: str | None = None
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    require_api_key: bool = False

    # --- Metadata DB --------------------------------------------------------
    metadata_db_url: str = "sqlite:///./metadata.sqlite3"

    # --- Observability ------------------------------------------------------
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_json: bool = True

    # --- Validators ---------------------------------------------------------
    @field_validator("embedding_dimension")
    @classmethod
    def _dimension_from_model(cls, v: int, info) -> int:
        """Keep dimension consistent with well-known OpenAI models."""
        model = info.data.get("embedding_model", "")
        if model == "text-embedding-3-large":
            return 3072
        if model == "text-embedding-3-small":
            return 1536
        return v

    @property
    def chroma_persist_path(self) -> str:
        return self.chroma_persist_dir

    @property
    def effective_score_threshold(self) -> float:
        """Score threshold adjusted for the active embedding backend.

        The PRD default of 0.72 is calibrated for OpenAI/HF semantic
        embeddings. The offline ``local`` hashing embeddings produce lower
        cosine scores, so a relaxed floor is used there to keep the offline
        path functional. Production (openai/huggingface) uses the configured
        ``score_threshold`` unchanged.
        """
        if self.embedding_provider == "local":
            return min(self.score_threshold, 0.1)
        return self.score_threshold

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in .env or switch "
                "EMBEDDING_PROVIDER/LLM_PROVIDER to 'local'."
            )
        return self.openai_api_key

    def require_pinecone(self) -> str:
        if not self.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Set it in .env or use "
                "VECTOR_STORE=chroma for local development."
            )
        return self.pinecone_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated :class:`Settings` instance."""
    return Settings()

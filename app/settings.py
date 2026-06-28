"""Centralised, validated configuration.

All env vars are loaded once at import time. Missing required vars raise a
clear ValidationError immediately, instead of failing mid-request later.

Usage:
    from app.settings import settings
    print(settings.groq_api_key.get_secret_value())
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required: provider credentials ---
    groq_api_key: SecretStr = Field(..., description="Groq API key for LLM completions")

    # --- Required: data plane ---
    database_url: PostgresDsn = Field(
        ...,
        description="Postgres + pgvector connection URL (SQLAlchemy-compatible)",
    )

    # --- Optional: Cloudflare R2  ---
    r2_endpoint_url: str | None = Field(default=None, description="R2 S3-compatible endpoint")
    r2_access_key_id: SecretStr | None = Field(default=None)
    r2_secret_access_key: SecretStr | None = Field(default=None)
    r2_bucket: str | None = Field(default=None)

    # --- Optional: Langfuse Cloud  ---
    langfuse_public_key: SecretStr | None = Field(default=None)
    langfuse_secret_key: SecretStr | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # --- Model + retrieval knobs ---
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384)
    reranker_model: str = Field(default="BAAI/bge-reranker-base")

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)
    top_k: int = Field(default=20, gt=0)
    rerank_k: int = Field(default=5, gt=0)
    min_similarity: float = Field(default=0.30, ge=0.0, le=1.0)

    # --- Runtime ---
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info: object) -> int:
        # Pydantic v2 cross-field validation: access via info.data
        data = getattr(info, "data", {})
        size = data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError(f"chunk_overlap ({v}) must be smaller than chunk_size ({size})")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — instantiates once, raises on missing vars."""
    return Settings()  # type: ignore[call-arg]  # values come from env


# Eager-load so import failures surface at module import, not first use.
settings = get_settings()

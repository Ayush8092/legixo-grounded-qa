"""Central application configuration.

Loads and validates application settings from environment variables and
.env using pydantic-settings.

This module is the single source of truth for configuration. Other modules
should use get_settings() rather than reading environment variables directly.

Secrets are never intentionally printed by this module.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================
    # API KEYS
    # ==========================================================

    gemini_api_key: SecretStr = Field(
        ...,
        validation_alias="GEMINI_API_KEY",
    )

    groq_api_key: SecretStr = Field(
        ...,
        validation_alias="GROQ_API_KEY",
    )

    pinecone_api_key: SecretStr = Field(
        ...,
        validation_alias="PINECONE_API_KEY",
    )

    # ==========================================================
    # PINECONE
    # ==========================================================

    pinecone_index_name: str = Field(
        default="legixo-grounded-qa",
        validation_alias="PINECONE_INDEX_NAME",
    )

    pinecone_namespace: str = Field(
        default="legixo-corpus",
        validation_alias="PINECONE_NAMESPACE",
    )

    pinecone_cloud: str = Field(
        default="aws",
        validation_alias="PINECONE_CLOUD",
    )

    pinecone_region: str = Field(
        default="us-east-1",
        validation_alias="PINECONE_REGION",
    )


    pinecone_ready_timeout_seconds: int = Field(
        default=120,
        validation_alias="PINECONE_READY_TIMEOUT_SECONDS",
    )
    # ==========================================================
    # EMBEDDINGS
    # ==========================================================

    embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="EMBEDDING_MODEL",
    )

    embedding_dimension: int = Field(
        default=3072,
        ge=1,
        validation_alias="EMBEDDING_DIMENSION",
    )

    embedding_max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        validation_alias="EMBEDDING_MAX_RETRIES",
        description="Bounded exponential-backoff retry attempts for transient Gemini errors (429 / RESOURCE_EXHAUSTED / 5xx).",
    )

    # ==========================================================
    # ANSWER LLM
    # ==========================================================

    answer_model: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias="ANSWER_MODEL",
    )

    answer_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias="ANSWER_TEMPERATURE",
    )

    groq_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias="GROQ_MAX_RETRIES",
        description="Built-in Groq SDK retry attempts for transient errors (429 / 5xx).",
    )

    # ==========================================================
    # RETRIEVAL
    # ==========================================================

    top_k: int = Field(
        default=10,
        ge=1,
        le=20,
        validation_alias="TOP_K",
    )

    score_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        validation_alias="SCORE_THRESHOLD",
    )

    query_fanout: int = Field(
        default=3,
        ge=1,
        le=5,
        validation_alias="QUERY_FANOUT",
    )

    # ==========================================================
    # RERANKING
    # ==========================================================

    rerank_enabled: bool = Field(
        default=True,
        validation_alias="RERANK_ENABLED",
        description=(
            "Rerank Pinecone's candidates (lexical BM25, no extra model/API) "
            "before grading. Set to false to fall back to the original "
            "retrieve -> grade flow, e.g. for debugging."
        ),
    )

    rerank_top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        validation_alias="RERANK_TOP_K",
        description=(
            "How many reranked chunks are kept for grading, out of the "
            "TOP_K Pinecone candidates. Should stay close to TOP_K on a "
            "small corpus — this trims low-relevance candidates, it isn't "
            "meant to aggressively cut recall (see docs/architecture.md)."
        ),
    )

    # ==========================================================
    # LANGGRAPH
    # ==========================================================

    max_retrieval_loops: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias="MAX_RETRIEVAL_LOOPS",
    )

    recursion_limit: int = Field(
        default=20,
        ge=5,
        le=100,
        validation_alias="RECURSION_LIMIT",
    )

    # ==========================================================
    # INGESTION
    # ==========================================================

    corpus_dir: str = Field(
        default="data/corpus",
        validation_alias="CORPUS_DIR",
        description="Immutable, shipped assignment corpus. POST /upload never writes here.",
    )

    upload_dir: str = Field(
        default="data/uploads",
        validation_alias="UPLOAD_DIR",
        description="Mutable, runtime knowledge base. POST /upload writes here; the immutable corpus_dir is untouched.",
    )

    max_upload_file_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB per file
        ge=1024,
        validation_alias="MAX_UPLOAD_FILE_SIZE_BYTES",
        description=(
            "Per-file size limit for POST /upload. Files are read up to "
            "this limit + 1 byte and rejected if that's exceeded, so an "
            "oversized upload is never fully loaded into memory."
        ),
    )

    # ==========================================================
    # API SERVER
    # ==========================================================

    api_host: str = Field(
        default="127.0.0.1",
        validation_alias="API_HOST",
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias="API_PORT",
    )

    api_reload: bool = Field(
        default=True,
        validation_alias="API_RELOAD",
    )

    include_trace: bool = Field(
        default=False,
        validation_alias="INCLUDE_TRACE",
    )

    service_init_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        validation_alias="SERVICE_INIT_TIMEOUT_SECONDS",
        description="Bounded wait for building the QAService (Gemini/Groq/Pinecone clients) on a request thread. Prevents /ready or /ask from hanging forever if a provider is unreachable.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
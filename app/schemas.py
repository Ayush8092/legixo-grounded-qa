"""Request/response models for the Q&A API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer.")


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    section: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]
    trace: list[str] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    detail: str


class UploadRejection(BaseModel):
    filename: str
    error: str


class UploadResponse(BaseModel):
    status: str  # "success" | "partial_success" | "error"
    files: list[str]
    rejected: list[UploadRejection] = []
    chunks_created: int = 0
    vectors_upserted: int = 0
    unchanged_chunks: int = 0
    stale_vectors_deleted: int = 0
    total_chunks: int = 0
    namespace_vector_count: int = 0

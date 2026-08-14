"""Offline unit tests for app.clients — embedding normalization, retry
backoff, and the Pinecone dimension-mismatch guard.

No network calls, no real API keys — the Gemini/Pinecone SDK objects are
replaced with small fakes.
"""

import math
from types import SimpleNamespace

import pytest

from app.clients import Embedder, _normalize, _validate_index_dimension
from app.config import Settings


def _settings(**overrides) -> Settings:
    base = dict(
        GEMINI_API_KEY="x",
        GROQ_API_KEY="x",
        PINECONE_API_KEY="x",
        EMBEDDING_DIMENSION=768,
        EMBEDDING_MAX_RETRIES=3,
    )
    base.update(overrides)
    return Settings(**base)


def test_normalize_produces_unit_length_vector():
    vec = _normalize([3.0, 4.0])  # 3-4-5 triangle -> length 5
    assert math.isclose(math.hypot(*vec), 1.0, rel_tol=1e-6)


def test_normalize_handles_zero_vector_without_dividing_by_zero():
    assert _normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbedding(v) for v in vectors]


class _FakeModels:
    """Stands in for `genai.Client().models`."""

    def __init__(self, vectors=None, fail_times: int = 0, exc: Exception | None = None):
        self._vectors = vectors or [[1.0, 2.0, 2.0]]
        self._fail_times = fail_times
        self._exc = exc
        self.calls: list[dict] = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._exc
        return _FakeEmbedResponse(self._vectors)


class _FakeGenaiClient:
    def __init__(self, models: _FakeModels):
        self.models = models


def _embedder_with_fake_client(models: _FakeModels, settings: Settings) -> Embedder:
    embedder = Embedder.__new__(Embedder)  # bypass __init__ (avoids real genai.Client())
    embedder._client = _FakeGenaiClient(models)
    embedder._model = settings.embedding_model
    embedder._dim = settings.embedding_dimension
    embedder._retries = settings.embedding_max_retries
    return embedder


def test_embed_documents_uses_retrieval_document_task_type():
    models = _FakeModels(vectors=[[3.0, 4.0]])
    embedder = _embedder_with_fake_client(models, _settings())
    embedder.embed_documents(["some chunk text"])
    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_embed_query_uses_retrieval_query_task_type():
    models = _FakeModels(vectors=[[3.0, 4.0]])
    embedder = _embedder_with_fake_client(models, _settings())
    embedder.embed_query("some question")
    assert models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_embeddings_are_normalized_to_unit_length():
    models = _FakeModels(vectors=[[3.0, 4.0]])
    embedder = _embedder_with_fake_client(models, _settings())
    vec = embedder.embed_query("q")
    assert math.isclose(math.hypot(*vec), 1.0, rel_tol=1e-6)


def test_embed_documents_mismatched_count_raises():
    models = _FakeModels(vectors=[[1.0, 0.0]])  # only 1 vector for 2 inputs
    embedder = _embedder_with_fake_client(models, _settings())
    with pytest.raises(RuntimeError):
        embedder.embed_documents(["text one", "text two"])


class _FakeIndexDescription:
    def __init__(self, dimension):
        self.dimension = dimension


class _FakePineconeClient:
    def __init__(self, dimension):
        self._dimension = dimension

    def describe_index(self, name):
        return _FakeIndexDescription(self._dimension)


def test_validate_index_dimension_passes_when_matching():
    settings = _settings(EMBEDDING_DIMENSION=768, PINECONE_INDEX_NAME="legixo-grounded-qa")
    pc = _FakePineconeClient(dimension=768)
    _validate_index_dimension(pc, settings)  # should not raise


def test_validate_index_dimension_raises_clear_error_on_mismatch():
    settings = _settings(EMBEDDING_DIMENSION=3072, PINECONE_INDEX_NAME="legixo-grounded-qa")
    pc = _FakePineconeClient(dimension=768)
    with pytest.raises(RuntimeError, match="768"):
        _validate_index_dimension(pc, settings)


def test_embed_retries_transient_server_error_then_succeeds(monkeypatch):
    from google.genai import errors as genai_errors

    monkeypatch.setattr("app.clients.time.sleep", lambda _seconds: None)
    server_error = genai_errors.ServerError(code=503, response_json={"error": {"message": "unavailable"}})
    models = _FakeModels(vectors=[[3.0, 4.0]], fail_times=2, exc=server_error)
    embedder = _embedder_with_fake_client(models, _settings(EMBEDDING_MAX_RETRIES=5))

    vec = embedder.embed_query("q")

    assert len(models.calls) == 3  # 2 failures + 1 success
    assert math.isclose(math.hypot(*vec), 1.0, rel_tol=1e-6)


def test_embed_gives_up_after_max_retries(monkeypatch):
    from google.genai import errors as genai_errors

    monkeypatch.setattr("app.clients.time.sleep", lambda _seconds: None)
    server_error = genai_errors.ServerError(code=503, response_json={"error": {"message": "unavailable"}})
    models = _FakeModels(vectors=[[3.0, 4.0]], fail_times=99, exc=server_error)
    embedder = _embedder_with_fake_client(models, _settings(EMBEDDING_MAX_RETRIES=3))

    with pytest.raises(genai_errors.ServerError):
        embedder.embed_query("q")

    assert len(models.calls) == 3


def test_embed_does_not_retry_non_transient_client_error(monkeypatch):
    from google.genai import errors as genai_errors

    monkeypatch.setattr("app.clients.time.sleep", lambda _seconds: None)
    client_error = genai_errors.ClientError(code=400, response_json={"error": {"message": "bad request"}})
    models = _FakeModels(vectors=[[3.0, 4.0]], fail_times=99, exc=client_error)
    embedder = _embedder_with_fake_client(models, _settings(EMBEDDING_MAX_RETRIES=5))

    with pytest.raises(genai_errors.ClientError):
        embedder.embed_query("q")

    assert len(models.calls) == 1  # no retry — not transient

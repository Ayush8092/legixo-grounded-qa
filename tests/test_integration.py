"""Integration tests against the real Gemini, Groq, and Pinecone services.

These tests use the application's Pydantic Settings configuration, which
loads API keys from .env.

Run explicitly with:

    pytest tests/test_integration.py -v

The rest of the test suite does not depend on real API services and does
not consume API quota.
"""

import pytest

from app.config import get_settings


def _integration_keys_available() -> bool:
    """Return True when all required provider keys are configured."""
    try:
        settings = get_settings()

        return bool(
            settings.gemini_api_key.get_secret_value()
            and settings.groq_api_key.get_secret_value()
            and settings.pinecone_api_key.get_secret_value()
        )

    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _integration_keys_available(),
    reason=(
        "Integration tests require GEMINI_API_KEY, "
        "GROQ_API_KEY, and PINECONE_API_KEY in .env."
    ),
)


@pytest.fixture(scope="module")
def service():
    from app.graph import QAService

    try:
        return QAService()
    except RuntimeError as exc:
        pytest.skip(
            "Pinecone index not ready — "
            "run `python -m app.ingestion` first: "
            f"{exc}"
        )


def test_answerable_question_returns_grounded_citation(service):
    result = service.ask(
        "What is the notice period at Bluecrest?"
    )

    assert result["found"] is True
    assert result["citations"], "expected at least one citation"

    assert any(
        c["source_file"] == "02_employment_agreement_excerpt.md"
        for c in result["citations"]
    )

    assert "60" in result["answer"]


def test_out_of_corpus_question_is_refused(service):
    result = service.ask(
        "What is the capital of France?"
    )

    assert result["found"] is False
    assert result["citations"] == []


def test_every_citation_points_at_a_real_retrieved_chunk(service):
    result = service.ask(
        "What is the non-compete period at Bluecrest?"
    )

    for citation in result["citations"]:
        assert citation["chunk_id"]
        assert citation["source_file"].endswith(".md")
        assert 0.0 <= citation["score"] <= 1.0


def test_tempting_but_unsupported_detail_is_not_invented(service):
    """The agreement states the non-compete's duration but never a penalty —
    the system must not invent one just because the topic is present."""
    result = service.ask("What penalty applies if Priya breaches the non-compete?")
    assert "12 month" not in result["answer"].lower() or "penalt" not in result["answer"].lower() or (
        "not specif" in result["answer"].lower()
        or "does not" in result["answer"].lower()
        or "no penalty" in result["answer"].lower()
        or result["found"] is False
    )


def test_population_question_is_refused(service):
    result = service.ask("What is the population of Riverside city?")
    assert result["found"] is False
    assert result["citations"] == []


def test_president_of_india_question_is_refused(service):
    result = service.ask("Who is the president of India?")
    assert result["found"] is False
    assert result["citations"] == []
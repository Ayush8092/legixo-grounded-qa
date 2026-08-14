"""Offline tests for the FastAPI layer (app.main).

`/health` never touches an external provider. `/ask` is tested with
`QAService` replaced by a fake, so these tests need no API keys and make
no network calls.
"""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings


class _FakeService:
    def __init__(self, result):
        self._result = result

    def ask(self, question: str):
        return self._result


@pytest.fixture(autouse=True)
def _reset_service():
    """Ensure the lazy-loaded singleton doesn't leak between tests."""
    main._service = None
    yield
    main._service = None


def _client():
    return TestClient(main.app)


def _isolated_settings(**overrides) -> Settings:
    """A Settings object independent of whatever the real environment/.env
    happens to have set. `get_settings()` is `@lru_cache`d at module scope,
    so any test that calls the *real* `get_settings()` without overriding
    it inherits whatever the process resolved first (e.g. a developer's
    real `INCLUDE_TRACE=true` in `.env` for local debugging) — tests that
    assert a specific default (like "trace is omitted when not requested")
    must not depend on that ambient state.
    """
    base = dict(GEMINI_API_KEY="x", GROQ_API_KEY="x", PINECONE_API_KEY="x")
    base.update(overrides)
    return Settings(**base)


def test_health_returns_ok_without_any_provider_setup():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_grounded_answer_with_citations(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: _isolated_settings(INCLUDE_TRACE=False))
    result = {
        "answer": "Either party may end the agreement by giving 60 days written notice.",
        "found": True,
        "citations": [
            {
                "chunk_id": "02_employment_agreement_excerpt::notice-period::0",
                "source_file": "02_employment_agreement_excerpt.md",
                "section": "Notice period",
                "snippet": "Either party may end this agreement...",
                "score": 0.83,
            }
        ],
        # The fake service still returns a populated trace (it's what the
        # real graph always produces internally) — the point of this test
        # is that /ask must drop it from the response when trace wasn't
        # requested and INCLUDE_TRACE is off, not that the service omits it.
        "trace": ["retrieve", "grade_chunks", "generate_answer", "validate_citations"],
    }
    monkeypatch.setattr(main, "get_service", lambda: _FakeService(result))

    response = _client().post("/ask", json={"question": "What is the notice period?"})
    body = response.json()

    assert response.status_code == 200
    assert body["found"] is True
    assert body["citations"][0]["source_file"] == "02_employment_agreement_excerpt.md"
    assert body.get("trace") is None  # trace not requested and INCLUDE_TRACE=False; field is omitted/null


def test_ask_with_trace_query_param_includes_trace(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: _isolated_settings(INCLUDE_TRACE=False))
    result = {
        "answer": "60 days written notice.",
        "found": True,
        "citations": [],
        "trace": ["retrieve", "grade_chunks"],
    }
    monkeypatch.setattr(main, "get_service", lambda: _FakeService(result))

    response = _client().post(
        "/ask", params={"trace": "true"}, json={"question": "What is the notice period?"}
    )
    body = response.json()
    assert body["trace"] == ["retrieve", "grade_chunks"]


def test_ask_includes_trace_when_include_trace_setting_is_enabled(monkeypatch):
    """The other side of the same toggle: INCLUDE_TRACE=true in settings
    should include the trace even without the ?trace=true query param."""
    monkeypatch.setattr(main, "get_settings", lambda: _isolated_settings(INCLUDE_TRACE=True))
    result = {
        "answer": "60 days written notice.",
        "found": True,
        "citations": [],
        "trace": ["retrieve", "grade_chunks"],
    }
    monkeypatch.setattr(main, "get_service", lambda: _FakeService(result))

    response = _client().post("/ask", json={"question": "What is the notice period?"})
    body = response.json()
    assert body["trace"] == ["retrieve", "grade_chunks"]


def test_ask_refusal_has_empty_citations(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: _isolated_settings(INCLUDE_TRACE=False))
    result = {
        "answer": "I cannot find the answer to this question in the provided documents.",
        "found": False,
        "citations": [],
        "trace": [],
    }
    monkeypatch.setattr(main, "get_service", lambda: _FakeService(result))

    response = _client().post("/ask", json={"question": "What is the capital of France?"})
    body = response.json()
    assert body["found"] is False
    assert body["citations"] == []


def test_ask_rejects_empty_question():
    response = _client().post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_surfaces_upstream_errors_as_502(monkeypatch):
    class _BrokenService:
        def ask(self, question):
            raise RuntimeError("Pinecone timeout")

    monkeypatch.setattr(main, "get_service", lambda: _BrokenService())
    response = _client().post("/ask", json={"question": "anything"})
    assert response.status_code in (502, 503)


def test_ready_reports_not_ready_when_service_init_fails(monkeypatch):
    def _boom():
        raise RuntimeError("Pinecone index 'legixo-grounded-qa' does not exist yet.")

    monkeypatch.setattr(main, "get_service", _boom)
    response = _client().get("/ready")
    body = response.json()
    assert response.status_code == 200  # /ready reports status in the body, not via HTTP code
    assert body["ready"] is False
    assert "does not exist" in body["detail"]


def test_ready_reports_ready_when_service_available(monkeypatch):
    monkeypatch.setattr(main, "get_service", lambda: _FakeService({}))
    response = _client().get("/ready")
    body = response.json()
    assert body["ready"] is True


def test_health_does_not_depend_on_service_availability(monkeypatch):
    """`/health` must stay green even if the QAService singleton is completely broken."""

    def _boom():
        raise RuntimeError("Pinecone is down")

    monkeypatch.setattr(main, "get_service", _boom)
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend_route_serves_html(monkeypatch):
    response = _client().get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

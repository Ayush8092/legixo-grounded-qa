"""Offline unit tests for the `rerank` node and its place in the graph.

Constructs `QAService` via `object.__new__` to skip `__init__` (which would
otherwise build real Gemini/Groq/Pinecone clients) — only `self.settings` is
set, which is all `_rerank` and `_build_graph` need. No network calls, no
API keys required beyond the placeholder strings `Settings` validates.
"""

from app.config import Settings
from app.graph import QAService


def _make_service(**settings_overrides) -> QAService:
    base = dict(GEMINI_API_KEY="x", GROQ_API_KEY="x", PINECONE_API_KEY="x")
    base.update(settings_overrides)
    service = object.__new__(QAService)
    service.settings = Settings(**base)
    return service


def _chunk(chunk_id, text, score=0.5):
    return {
        "chunk_id": chunk_id,
        "source_file": chunk_id.split("::")[0] + ".md",
        "section": "x",
        "text": text,
        "score": score,
    }


def test_rerank_node_reorders_chunks_by_lexical_relevance():
    service = _make_service(RERANK_ENABLED=True, RERANK_TOP_K=2)
    state = {
        "question": "When is the next hearing for Arvind Mehta v. Northfield Logistics?",
        "chunks": [
            _chunk("b::x::0", "Unrelated content about something else entirely.", score=0.6),
            _chunk("a::x::0", "Arvind Mehta v. Northfield Logistics: next hearing 15 August 2025.", score=0.5),
        ],
    }

    result = service._rerank(state)

    assert result["chunks"][0]["chunk_id"] == "a::x::0"
    assert len(result["chunks"]) == 2
    assert any("rerank:" in t for t in result["trace"])


def test_rerank_node_respects_configured_top_k():
    service = _make_service(RERANK_ENABLED=True, RERANK_TOP_K=1)
    state = {
        "question": "notice period",
        "chunks": [
            _chunk("a::x::0", "notice period is 60 days"),
            _chunk("b::x::0", "notice period clause details here too"),
            _chunk("c::x::0", "completely unrelated rent and deposit terms"),
        ],
    }

    result = service._rerank(state)

    assert len(result["chunks"]) == 1


def test_rerank_node_disabled_is_a_passthrough_that_does_not_touch_chunks():
    service = _make_service(RERANK_ENABLED=False)
    original_chunks = [_chunk("a::x::0", "content")]
    state = {"question": "q", "chunks": original_chunks}

    result = service._rerank(state)

    # A passthrough returns no "chunks" key at all, so LangGraph's state
    # merge leaves state["chunks"] exactly as retrieve() produced it.
    assert "chunks" not in result
    assert any("skipped" in t.lower() for t in result["trace"])


def test_rerank_node_with_no_candidates_is_a_safe_passthrough():
    service = _make_service(RERANK_ENABLED=True)
    result = service._rerank({"question": "q", "chunks": []})
    assert "chunks" not in result


def test_graph_execution_visits_rerank_between_retrieve_and_grade_chunks():
    """End-to-end execution-order check via graph.invoke() (a stable,
    version-independent LangGraph API) rather than introspecting the
    compiled graph's internal node/edge representation, which varies more
    across langgraph versions. Every node is stubbed so this needs no real
    provider clients and completes in one pass (grade reports sufficient
    immediately, so rewrite_query/retrieve never re-run)."""
    service = _make_service(RERANK_ENABLED=True, RERANK_TOP_K=5)
    visited: list[str] = []

    def fake_retrieve(state):
        visited.append("retrieve")
        return {
            "chunks": [_chunk("a::x::0", "some content")],
            "trace": ["retrieve: 1 chunk"],
        }

    def fake_grade(state):
        visited.append("grade_chunks")
        return {
            "grade": {"sufficient": True, "relevant_chunk_ids": ["a::x::0"], "reason": "ok"},
            "trace": ["grade_chunks: sufficient"],
        }

    def fake_generate(state):
        visited.append("generate_answer")
        return {
            "found": True,
            "answer": "the answer",
            "citations": ["a::x::0"],
            "trace": ["generate_answer: found=True"],
        }

    def fake_validate(state):
        visited.append("validate_citations")
        return {
            "citations": [{"chunk_id": "a::x::0", "source_file": "a.md", "section": "x"}],
            "trace": ["validate_citations: 1 valid, 0 dropped"],
        }

    real_rerank = service._rerank

    def tracking_rerank(state):
        visited.append("rerank")
        return real_rerank(state)

    service._retrieve = fake_retrieve
    service._rerank = tracking_rerank
    service._grade_chunks = fake_grade
    service._generate_answer = fake_generate
    service._validate_citations = fake_validate

    graph = service._build_graph()
    graph.invoke(
        {
            "question": "q",
            "queries": [],
            "chunks": [],
            "grade": {},
            "loops": 0,
            "answer": "",
            "found": False,
            "citations": [],
            "trace": [],
        },
        config={"recursion_limit": 20},
    )

    assert visited == ["retrieve", "rerank", "grade_chunks", "generate_answer", "validate_citations"]

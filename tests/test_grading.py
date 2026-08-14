"""Offline unit tests for app.llm — grading, rewriting, answer parsing.

The real Groq client is replaced with a tiny fake that mimics the shape of
`groq.Groq().chat.completions.create(...)` and returns a canned JSON reply,
so these tests exercise our parsing/validation logic without any network
call or API key.
"""

import json

import groq
import pytest

from app.llm import (
    REFUSAL_TEXT,
    LLMProviderError,
    generate_answer,
    grade_chunks,
    rewrite_query,
)

MODEL = "llama-3.1-8b-instant"

CHUNKS = [
    {
        "chunk_id": "02_employment_agreement_excerpt::notice-period::0",
        "source_file": "02_employment_agreement_excerpt.md",
        "section": "Notice period",
        "text": "Either party may end this agreement by giving 60 days written notice.",
        "score": 0.83,
    }
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, reply: str, raise_exc: Exception | None = None):
        self._reply = reply
        self._raise = raise_exc
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise:
            raise self._raise
        return _FakeCompletion(self._reply)


class _FakeChat:
    """Stand-in for `groq.Groq()`: same `.chat.completions.create(...)` shape."""

    def __init__(self, reply: str = "{}", raise_exc: Exception | None = None):
        self.chat = type("_C", (), {})()
        self.chat.completions = _FakeCompletions(reply, raise_exc)


class _FakeProviderError(groq.APIError):
    """A real `groq.APIError` subclass, built without needing the SDK's own
    constructor args (httpx.Request, etc.) — just enough to prove
    `isinstance(exc, groq.APIError)` is True, which is what `_chat_json`
    actually branches on. Stands in for RateLimitError / APITimeoutError /
    APIConnectionError / InternalServerError, all of which share this base.
    """

    def __init__(self, message: str):
        Exception.__init__(self, message)


def test_grade_chunks_with_no_chunks_is_always_insufficient():
    chat = _FakeChat("should not be called")
    grade = grade_chunks(chat, MODEL, "irrelevant question", [])
    assert grade["sufficient"] is False
    assert grade["relevant_chunk_ids"] == []


def test_grade_chunks_parses_sufficient_response():
    reply = json.dumps(
        {
            "relevant_labels": ["CANDIDATE_1"],
            "reason": "directly states the notice period",
        }
    )
    chat = _FakeChat(reply)
    grade = grade_chunks(chat, MODEL, "What is the notice period?", CHUNKS)
    assert grade["sufficient"] is True
    assert grade["relevant_chunk_ids"] == ["02_employment_agreement_excerpt::notice-period::0"]


def test_grade_chunks_requests_json_object_response_format():
    reply = json.dumps({"relevant_labels": [], "reason": "x"})
    chat = _FakeChat(reply)
    grade_chunks(chat, MODEL, "question", CHUNKS)
    assert chat.chat.completions.last_kwargs["response_format"] == {"type": "json_object"}
    assert chat.chat.completions.last_kwargs["model"] == MODEL


def test_grade_chunks_drops_relevant_labels_not_actually_shown():
    """A grader that hallucinates a label (or a chunk_id) that was never
    actually shown for this call must not make it into the result."""
    reply = json.dumps(
        {"relevant_labels": ["CANDIDATE_99"], "reason": "x"}
    )
    chat = _FakeChat(reply)
    grade = grade_chunks(chat, MODEL, "question", CHUNKS)
    assert grade["relevant_chunk_ids"] == []
    assert grade["sufficient"] is False  # sufficient requires >=1 real relevant id


def test_grade_chunks_drops_a_fabricated_sibling_chunk_id_that_was_never_retrieved():
    """Regression test for the live grading failure (lexigo_project_improve
    .docx): shown only one candidate (CANDIDATE_1, whose real chunk_id ends
    in `::header::0`), an 8B-class grader would sometimes extrapolate
    sibling chunk_ids like `...header::1` / `...header::2` that were never
    actually retrieved, rather than returning the CANDIDATE_1 label it was
    shown. Those fabricated raw chunk_ids must still be filtered out even
    though the model bypassed the label contract entirely."""
    reply = json.dumps(
        {
            "relevant_labels": [
                "uploads::insurance_test.txt::header::1",
                "uploads::insurance_test.txt::header::2",
            ],
            "reason": "mentions the dental insurance premium percentage",
        }
    )
    chat = _FakeChat(reply)
    grade = grade_chunks(chat, MODEL, "question", CHUNKS)
    assert grade["relevant_chunk_ids"] == []
    assert grade["sufficient"] is False


def test_grade_chunks_ignores_a_models_own_sufficient_flag_when_it_disagrees_with_relevance():
    """Regression test for the eval-16 (partial-information) bug: a smaller
    model sometimes selects the correct on-topic chunk as relevant, but then
    separately (and wrongly) reports "sufficient": false because one
    specific requested detail (e.g. a penalty) isn't in that chunk.
    `sufficient` must be derived from relevant_chunk_ids alone, not from the
    model's own redundant self-assessment, so this case still routes to
    generate_answer instead of a false refusal."""
    reply = json.dumps(
        {
            "sufficient": False,
            "relevant_labels": ["CANDIDATE_1"],
            "reason": "states the notice period topic but not every possible detail",
        }
    )
    chat = _FakeChat(reply)
    grade = grade_chunks(chat, MODEL, "some partial-information question", CHUNKS)
    assert grade["sufficient"] is True
    assert grade["relevant_chunk_ids"] == ["02_employment_agreement_excerpt::notice-period::0"]


def test_grade_chunks_selects_relevant_chunks_across_multiple_documents():
    """Regression test for the eval-09 (multi-document) shape: the grader
    must be able to select relevant chunks from more than one source_file
    for a single question."""
    multi_doc_chunks = CHUNKS + [
        {
            "chunk_id": "01_matter_memo_arvind_v_northfield::next-hearing::0",
            "source_file": "01_matter_memo_arvind_v_northfield.md",
            "section": "Next hearing",
            "text": "15 August 2025 — witness for the plaintiff to be examined.",
            "score": 0.61,
        },
        {
            "chunk_id": "06_property_lease_clause::subletting::0",
            "source_file": "06_property_lease_clause.md",
            "section": "Subletting",
            "text": "Subletting requires written landlord consent.",
            "score": 0.58,
        },
    ]
    reply = json.dumps(
        {
            "relevant_labels": ["CANDIDATE_1", "CANDIDATE_2"],
            "reason": "notice period chunk and the next hearing date/witness chunk are both relevant",
        }
    )
    chat = _FakeChat(reply)
    grade = grade_chunks(chat, MODEL, "multi-document question", multi_doc_chunks)
    assert grade["sufficient"] is True
    assert set(grade["relevant_chunk_ids"]) == {
        "02_employment_agreement_excerpt::notice-period::0",
        "01_matter_memo_arvind_v_northfield::next-hearing::0",
    }
    # The unrelated lease chunk must not be pulled in just because it was retrieved.
    assert "06_property_lease_clause::subletting::0" not in grade["relevant_chunk_ids"]


def test_grade_chunks_selects_the_one_chunk_shown_when_model_correctly_uses_its_label():
    """The live health-insurance case from the bug report: a single small
    chunk covers the whole short uploaded document (health AND dental
    insurance together) as CANDIDATE_1. When the model correctly returns
    that label, the real chunk_id must come through as relevant."""
    insurance_chunk = [
        {
            "chunk_id": "uploads::insurance_test.txt::header::0",
            "source_file": "insurance_test.txt",
            "section": "header",
            "text": (
                "Health Insurance\n\nThe company pays 80% of the health "
                "insurance premium.\n\nDental Insurance\n\nThe company pays "
                "50% of the dental insurance premium."
            ),
            "score": 0.91,
        }
    ]
    reply = json.dumps(
        {
            "relevant_labels": ["CANDIDATE_1"],
            "reason": "states the dental insurance premium percentage",
        }
    )
    chat = _FakeChat(reply)
    grade = grade_chunks(
        chat, MODEL, "What percentage of the dental insurance premium does the company pay?", insurance_chunk
    )
    assert grade["sufficient"] is True
    assert grade["relevant_chunk_ids"] == ["uploads::insurance_test.txt::header::0"]


def test_grade_chunks_handles_malformed_json_as_insufficient():
    chat = _FakeChat("not json at all")
    grade = grade_chunks(chat, MODEL, "question", CHUNKS)
    assert grade["sufficient"] is False
    assert grade["relevant_chunk_ids"] == []


def test_grade_chunks_handles_provider_exception_as_insufficient():
    """A generic/unexpected exception (NOT a real Groq/provider error) must
    not crash the graph — it degrades to insufficient. This is the fallback
    for "something unexpected went wrong in the call path", distinct from a
    confirmed provider outage (see the next test)."""
    chat = _FakeChat(raise_exc=RuntimeError("connection reset"))
    grade = grade_chunks(chat, MODEL, "question", CHUNKS)
    assert grade["sufficient"] is False
    assert grade["relevant_chunk_ids"] == []


def test_grade_chunks_propagates_real_provider_error_instead_of_masking_it():
    """A genuine `groq.APIError` (rate limit, timeout, connection, 5xx) must
    propagate as `LLMProviderError`, NOT degrade into a fake 'insufficient
    evidence' grade. This is the root cause of the live failure where a
    correctly-retrieved chunk was refused: a Groq 429 was previously
    swallowed into `{}` -> sufficient=False, wasting a retry loop and then
    producing a false refusal instead of surfacing as a 503."""
    chat = _FakeChat(raise_exc=_FakeProviderError("rate limit reached"))
    with pytest.raises(LLMProviderError):
        grade_chunks(chat, MODEL, "question", CHUNKS)


def test_generate_answer_propagates_real_provider_error_instead_of_masking_it():
    chat = _FakeChat(raise_exc=_FakeProviderError("rate limit reached"))
    with pytest.raises(LLMProviderError):
        generate_answer(chat, MODEL, "question", CHUNKS)


def test_rewrite_query_returns_bounded_number_of_queries():
    reply = json.dumps({"queries": ["a", "b", "c", "d", "e"]})
    chat = _FakeChat(reply)
    queries = rewrite_query(chat, MODEL, "original question", "original question", fanout=3)
    assert len(queries) == 3


def test_rewrite_query_deduplicates_case_insensitively():
    reply = json.dumps(
        {"queries": ["Notice Period", "notice period", " notice period ", "termination clause"]}
    )
    chat = _FakeChat(reply)
    queries = rewrite_query(chat, MODEL, "q", "q", fanout=5)
    assert queries == ["Notice Period", "termination clause"]


def test_rewrite_query_drops_blank_entries():
    reply = json.dumps({"queries": ["", "   ", "real query"]})
    chat = _FakeChat(reply)
    queries = rewrite_query(chat, MODEL, "q", "q", fanout=5)
    assert queries == ["real query"]


def test_rewrite_query_falls_back_to_original_question_on_empty_reply():
    chat = _FakeChat(json.dumps({"queries": []}))
    queries = rewrite_query(chat, MODEL, "original question", "original question", fanout=3)
    assert queries == ["original question"]


def test_generate_answer_parses_found_true():
    reply = json.dumps(
        {
            "found": True,
            "answer": "60 days written notice.",
            "evidence_refs": ["EVIDENCE_1"],
        }
    )
    chat = _FakeChat(reply)
    result = generate_answer(chat, MODEL, "What is the notice period?", CHUNKS)
    assert result["found"] is True
    assert result["answer"] == "60 days written notice."
    assert result["cited_chunk_ids"] == ["02_employment_agreement_excerpt::notice-period::0"]


def test_generate_answer_defaults_to_refusal_on_unparseable_reply():
    chat = _FakeChat("garbage, not json")
    result = generate_answer(chat, MODEL, "question", CHUNKS)
    assert result["found"] is False
    assert result["answer"] == REFUSAL_TEXT
    assert result["cited_chunk_ids"] == []


def test_generate_answer_states_partial_info_without_inventing_missing_detail():
    """The model is instructed to answer the supported portion and flag the
    missing detail rather than refuse or hallucinate — this test only checks
    our parsing passes that through; the prompt wording itself is exercised
    live in tests/test_integration.py."""
    reply = json.dumps(
        {
            "found": True,
            "answer": (
                "The agreement states the non-compete lasts 12 months after "
                "leaving; it does not specify any penalty for breaching it."
            ),
            "evidence_refs": ["EVIDENCE_1"],
        }
    )
    chat = _FakeChat(reply)
    result = generate_answer(chat, MODEL, "What penalty applies for breaching?", CHUNKS)
    assert result["found"] is True
    assert "12 months" in result["answer"]
    assert "not specify" in result["answer"] or "not specified" in result["answer"]
    assert result["cited_chunk_ids"] == ["02_employment_agreement_excerpt::notice-period::0"]


def test_generate_answer_passes_through_configured_temperature():
    reply = json.dumps({"found": True, "answer": "x", "evidence_refs": []})
    chat = _FakeChat(reply)
    generate_answer(chat, MODEL, "question", CHUNKS, temperature=0.2)
    assert chat.chat.completions.last_kwargs["temperature"] == 0.2


def test_generate_answer_maps_evidence_label_to_exact_original_chunk_id():
    """The core fix (final_upgrade_lexigo.docx #1): the model never sees or
    returns a chunk_id, only an EVIDENCE_N label — application code maps it
    back to the exact original chunk_id. This is what makes citation
    selection deterministic regardless of whether the model would have
    reproduced a real chunk_id string correctly."""
    two_chunks = CHUNKS + [
        {
            "chunk_id": "01_matter_memo_arvind_v_northfield::next-hearing::0",
            "source_file": "01_matter_memo_arvind_v_northfield.md",
            "section": "Next hearing",
            "text": "15 August 2025 - witness for the plaintiff (billing head) to be examined.",
            "score": 0.79,
        }
    ]
    reply = json.dumps(
        {
            "found": True,
            "answer": "The witness for the plaintiff is to be examined on 15 August 2025.",
            "evidence_refs": ["EVIDENCE_2"],
        }
    )
    chat = _FakeChat(reply)
    result = generate_answer(chat, MODEL, "What happens at the next hearing?", two_chunks)
    assert result["cited_chunk_ids"] == ["01_matter_memo_arvind_v_northfield::next-hearing::0"]


def test_generate_answer_drops_an_evidence_ref_that_was_not_actually_shown():
    """A hallucinated/mistyped/stale label (e.g. 'EVIDENCE_99', or the model
    reverting to writing out a raw chunk_id despite instructions) must never
    reach citation validation as if it were a real chunk_id."""
    reply = json.dumps(
        {
            "found": True,
            "answer": "x",
            "evidence_refs": [
                "EVIDENCE_99",
                "02_employment_agreement_excerpt::notice-period::0",  # a raw chunk_id, not a label
            ],
        }
    )
    chat = _FakeChat(reply)
    result = generate_answer(chat, MODEL, "question", CHUNKS)
    assert result["cited_chunk_ids"] == []


def test_generate_answer_can_cite_a_subset_of_the_evidence_shown():
    """The grader may surface multiple relevant chunks; the answer only
    needs to cite the ones actually used, not every chunk shown."""
    three_chunks = CHUNKS + [
        {
            "chunk_id": "b::x::0",
            "source_file": "b.md",
            "section": "x",
            "text": "unrelated but graded-relevant content",
            "score": 0.6,
        },
        {
            "chunk_id": "c::x::0",
            "source_file": "c.md",
            "section": "x",
            "text": "also unrelated but graded-relevant content",
            "score": 0.55,
        },
    ]
    reply = json.dumps({"found": True, "answer": "60 days.", "evidence_refs": ["EVIDENCE_1"]})
    chat = _FakeChat(reply)
    result = generate_answer(chat, MODEL, "What is the notice period?", three_chunks)
    assert result["cited_chunk_ids"] == ["02_employment_agreement_excerpt::notice-period::0"]

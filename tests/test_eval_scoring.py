"""Offline unit tests for eval.run_eval.score_case — the allowed/required
source-file distinction and the strengthened eval-16-style semantic checks.

No network calls, no API keys, no live graph — `result` dicts are
hand-built to look like `QAService.ask(...)` output.
"""

from eval.run_eval import score_case


def _case(**overrides):
    base = {
        "id": "t-1",
        "question": "q",
        "category": "answerable_direct",
        "answerable": True,
        "allowed_source_files": ["a.md"],
        "expected_keywords": [],
    }
    base.update(overrides)
    return base


def _result(**overrides):
    base = {"found": True, "answer": "", "citations": []}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------
# allowed_source_files (ceiling) vs required_source_files (floor)
# ---------------------------------------------------------------------


def test_single_allowed_file_cited_passes():
    case = _case(allowed_source_files=["a.md"])
    result = _result(citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}])
    assert score_case(case, result)["citation_ok"] is True


def test_citation_outside_allowed_set_fails():
    case = _case(allowed_source_files=["a.md"])
    result = _result(citations=[{"source_file": "b.md", "chunk_id": "b.md::x::0"}])
    assert score_case(case, result)["citation_ok"] is False


def test_multi_document_case_passes_with_only_one_of_two_allowed_files_cited():
    """The assignment explicitly says not to force multiple citations when
    one document fully answers the question (eval-09's shape): allowed can
    list two files while required stays unset, so a single valid citation
    from either still passes."""
    case = _case(allowed_source_files=["a.md", "b.md"])  # no required_source_files
    result = _result(citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}])
    assert score_case(case, result)["citation_ok"] is True


def test_required_source_file_missing_from_citations_fails():
    case = _case(allowed_source_files=["a.md", "b.md"], required_source_files=["b.md"])
    result = _result(citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}])
    assert score_case(case, result)["citation_ok"] is False


def test_required_source_file_present_passes():
    case = _case(allowed_source_files=["a.md", "b.md"], required_source_files=["b.md"])
    result = _result(
        citations=[
            {"source_file": "a.md", "chunk_id": "a.md::x::0"},
            {"source_file": "b.md", "chunk_id": "b.md::y::0"},
        ]
    )
    assert score_case(case, result)["citation_ok"] is True


def test_legacy_expected_source_files_key_still_works():
    """Backward compatibility: a case written with the old key name (before
    this change) must keep working without modification."""
    case = _case(expected_source_files=["a.md"])
    del case["allowed_source_files"]
    result = _result(citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}])
    assert score_case(case, result)["citation_ok"] is True


def test_no_citations_at_all_fails_even_if_allowed_set_is_satisfied_vacuously():
    case = _case(allowed_source_files=["a.md"])
    result = _result(citations=[])
    assert score_case(case, result)["citation_ok"] is False


# ---------------------------------------------------------------------
# eval-16-style strengthened semantic checks
# ---------------------------------------------------------------------


def test_correct_partial_information_answer_passes():
    case = _case(
        expected_keywords=["12 months"],
        any_of_keywords=["not specif", "no penalty", "not stated"],
        forbidden_keywords=["\u20b9", "terminat"],
    )
    result = _result(
        answer=(
            "The agreement states the non-compete lasts 12 months after "
            "leaving; it does not specify any penalty for breaching it."
        ),
        citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}],
    )
    scored = score_case(case, result)
    assert scored["answer_ok"] is True
    assert scored["passed"] is True


def test_plausible_but_wrong_answer_reusing_the_duration_as_the_penalty_fails():
    """Exactly the adversarial example from improvement_010: an answer that
    contains the expected keyword '12 months' but never actually says the
    penalty is unspecified must NOT pass."""
    case = _case(
        expected_keywords=["12 months"],
        any_of_keywords=["not specif", "no penalty", "not stated"],
    )
    result = _result(
        answer="The penalty is 12 months.",
        citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}],
    )
    scored = score_case(case, result)
    assert scored["answer_ok"] is False


def test_invented_monetary_penalty_fails_forbidden_keyword_check():
    case = _case(
        expected_keywords=["12 months"],
        any_of_keywords=["not specif"],
        forbidden_keywords=["\u20b9", "fine of", "terminat"],
    )
    result = _result(
        answer=(
            "The non-compete lasts 12 months; breaching it incurs a "
            "\u20b950,000 fine."
        ),
        citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}],
    )
    scored = score_case(case, result)
    assert scored["answer_ok"] is False


def test_cases_without_any_of_or_forbidden_keywords_are_unaffected():
    """Every other existing eval case has no any_of_keywords/forbidden_keywords
    at all — those checks must be vacuously true and not change behavior."""
    case = _case(expected_keywords=["60 days"])
    result = _result(
        answer="The notice period is 60 days.",
        citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}],
    )
    scored = score_case(case, result)
    assert scored["answer_ok"] is True
    assert scored["passed"] is True


def test_unanswerable_case_still_requires_no_citations_and_found_false():
    case = _case(answerable=False, allowed_source_files=[])
    result = _result(found=False, answer="I cannot find...", citations=[])
    scored = score_case(case, result)
    assert scored["passed"] is True

    bad_result = _result(found=False, answer="I cannot find...", citations=[{"source_file": "a.md", "chunk_id": "a.md::x::0"}])
    assert score_case(case, bad_result)["citation_ok"] is False

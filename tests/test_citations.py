"""Offline unit tests for app.llm.validate_citations (the citation guard).

No network calls, no API keys required — this is a pure function.
"""

from app.llm import validate_citations

RETRIEVED = [
    {
        "chunk_id": "02_employment_agreement_excerpt::notice-period::0",
        "source_file": "02_employment_agreement_excerpt.md",
        "section": "Notice period",
        "text": "Either party may end this agreement by giving 60 days written notice.",
        "score": 0.83,
    },
    {
        "chunk_id": "02_employment_agreement_excerpt::non-compete::0",
        "source_file": "02_employment_agreement_excerpt.md",
        "section": "Non-compete",
        "text": "For 12 months after leaving, the employee may not work for a competitor.",
        "score": 0.71,
    },
]


def test_valid_citation_passes_through():
    result = validate_citations(
        ["02_employment_agreement_excerpt::notice-period::0"], RETRIEVED
    )
    assert len(result) == 1
    assert result[0]["source_file"] == "02_employment_agreement_excerpt.md"
    assert result[0]["section"] == "Notice period"
    assert result[0]["score"] == 0.83


def test_fabricated_chunk_id_is_dropped():
    result = validate_citations(["does-not-exist::fake::0"], RETRIEVED)
    assert result == []


def test_chunk_id_not_retrieved_this_request_is_dropped():
    """Level 1: a real chunk_id from a DIFFERENT request must not validate here."""
    result = validate_citations(
        ["06_property_lease_clause::rent::0"], RETRIEVED
    )
    assert result == []


def test_duplicate_citations_are_deduplicated():
    result = validate_citations(
        [
            "02_employment_agreement_excerpt::notice-period::0",
            "02_employment_agreement_excerpt::notice-period::0",
        ],
        RETRIEVED,
    )
    assert len(result) == 1


def test_mixed_valid_and_fabricated_keeps_only_valid():
    result = validate_citations(
        [
            "02_employment_agreement_excerpt::notice-period::0",
            "fabricated::section::9",
        ],
        RETRIEVED,
    )
    assert len(result) == 1
    assert result[0]["chunk_id"] == "02_employment_agreement_excerpt::notice-period::0"


def test_empty_citation_list_returns_empty():
    assert validate_citations([], RETRIEVED) == []


def test_no_retrieved_chunks_means_nothing_can_validate():
    assert validate_citations(["anything::x::0"], []) == []


def test_snippet_is_truncated_and_metadata_matches_source_chunk():
    """Level 2: citation fields are taken from the retrieved chunk's own
    metadata, never from anything the model said."""
    result = validate_citations(
        ["02_employment_agreement_excerpt::non-compete::0"], RETRIEVED
    )
    assert result[0]["section"] == "Non-compete"
    assert result[0]["snippet"].startswith("For 12 months")

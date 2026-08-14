"""Offline unit tests for app.reranker — pure Python, no network, no LLM.

Test actual reranking behavior (ordering, multi-document retention,
out-of-corpus passthrough) rather than just "was it called" — see
final_improvement_lexido.docx section 20: "Do not write tests that merely
assert reranker_called == True."
"""

from app.reranker import rerank


def _chunk(chunk_id, text, source_file=None, score=0.5):
    return {
        "chunk_id": chunk_id,
        "source_file": source_file or chunk_id.split("::")[0] + ".md",
        "section": "x",
        "text": text,
        "score": score,
    }


def test_rerank_promotes_lexically_relevant_chunk_over_vector_favored_one():
    """A chunk Pinecone ranked lower (by vector score) but that shares far
    more literal terms with the question should end up ranked ABOVE a
    chunk that merely shares one word/topic — the exact failure pattern
    documented for eval-09 (dense retrieval under-ranking an entity-rich,
    lexically obvious match)."""
    chunks = [
        _chunk(
            "b::x::0",
            "Document: Some Other Matter. Section: misc. General legal text "
            "with only a passing, unrelated mention.",
            score=0.60,  # Pinecone ranked this HIGHER by vector score
        ),
        _chunk(
            "a::next-hearing::0",
            "Document: Internal memo - Arvind Mehta v. Northfield Logistics. "
            "Section: Next hearing. 15 August 2025 - witness for the "
            "plaintiff to be examined.",
            score=0.55,  # Pinecone ranked this LOWER by vector score
        ),
    ]
    question = "When is the next hearing in Arvind Mehta v. Northfield Logistics, and what happens at it?"

    result = rerank(question, chunks, top_k=2)

    assert result[0]["chunk_id"] == "a::next-hearing::0"
    assert result[0]["rerank_score"] > result[1]["rerank_score"]


def test_rerank_top_k_truncates_to_requested_count():
    chunks = [_chunk(f"c{i}::x::0", f"chunk number {i} about rent and deposit") for i in range(5)]
    result = rerank("rent and deposit", chunks, top_k=2)
    assert len(result) == 2


def test_rerank_preserves_multiple_relevant_documents():
    """Multi-document questions must keep chunks from more than one source
    document in the reranked output when both are genuinely relevant —
    reranking must not collapse retrieval down to a single source."""
    chunks = [
        _chunk(
            "01_matter_memo::next-hearing::0",
            "Arvind Mehta v. Northfield Logistics: 15 August 2025, witness for the plaintiff to be examined.",
            source_file="01_matter_memo.md",
        ),
        _chunk(
            "03_hearing_notice::today::0",
            "Case Arvind Mehta v. Northfield Logistics: arguments on invoice set-off, time 11:00 a.m.",
            source_file="03_hearing_notice.md",
        ),
        _chunk(
            "06_lease::subletting::0",
            "Subletting requires written landlord consent for the Harbor View unit.",
            source_file="06_lease.md",
        ),
    ]
    question = "When is the next hearing in Arvind Mehta v. Northfield Logistics and what is scheduled?"

    result = rerank(question, chunks, top_k=3)
    result_sources = {c["source_file"] for c in result[:2]}

    assert "01_matter_memo.md" in result_sources
    assert "03_hearing_notice.md" in result_sources


def test_rerank_with_completely_disjoint_vocabulary_gives_zero_scores_and_preserves_order():
    """The genuinely-zero-overlap case: an out-of-corpus question built
    from vocabulary that cannot appear in any real chunk (not even a
    stopword), so BM25 has nothing at all to score against. This is the
    property reranking must guarantee: no crash, no fabricated relevance,
    and — since Python's sort is stable — the original (Pinecone
    vector-similarity) order is preserved so the grader sees the same
    candidates it always would have, unranked by reranking."""
    chunks = [
        _chunk("06_lease::rent::0", "Monthly rent is $2,000, due on the first of the month."),
        _chunk("02_employment::notice::0", "Either party may end this agreement with 60 days notice."),
    ]
    # Deliberately not a real English question — invented tokens guarantee
    # zero shared vocabulary with the chunks above, including stopwords,
    # which a realistic question like "Who is the president of India?"
    # does NOT guarantee (see the test below).
    question = "Zorvathi plimquex nendrable?"

    result = rerank(question, chunks, top_k=5)

    assert len(result) == 2
    assert [c["chunk_id"] for c in result] == [c["chunk_id"] for c in chunks]
    assert all(c["rerank_score"] == 0 for c in result)


def test_rerank_semantically_unrelated_question_sharing_only_stopwords_does_not_crash():
    """A realistic out-of-corpus question ("Who is the president of
    India?") legitimately shares common stopwords ("is", "the", "of") with
    unrelated chunks — the BM25 tokenizer does not strip stopwords, and
    per the assignment feedback this is intentional/acceptable: fixing
    this by redesigning the BM25 implementation (e.g. adding a stopword
    list) is explicitly out of scope here. This test only asserts what
    actually matters for correctness: reranking must not crash, must not
    drop or invent a chunk, and every chunk's `rerank_score` must be a
    valid non-negative number — NOT that every score is exactly 0, which
    doesn't hold once stopword overlap is allowed. Whether a
    semantically-unrelated-but-lexically-noisy candidate is ultimately
    rejected is the grader's job (see app/llm.py's grader prompt and
    tests/test_grading.py), not the reranker's."""
    chunks = [
        _chunk("06_lease::rent::0", "Monthly rent is $2,000, due on the first of the month."),
        _chunk("02_employment::notice::0", "Either party may end this agreement with 60 days notice."),
    ]
    question = "Who is the president of India?"

    result = rerank(question, chunks, top_k=5)

    assert len(result) == 2
    assert {c["chunk_id"] for c in result} == {c["chunk_id"] for c in chunks}
    assert all(isinstance(c["rerank_score"], (int, float)) and c["rerank_score"] >= 0 for c in result)


def test_rerank_preserves_stable_order_for_exactly_tied_scores():
    """When multiple chunks end up with the exact same BM25 score (the tie
    case — trivially true when none of them share any token with the
    query), Python's stable sort must preserve their original relative
    (Pinecone vector-similarity) order rather than reordering ties
    arbitrarily."""
    chunks = [
        _chunk("a::x::0", "First unrelated chunk about rent and deposits."),
        _chunk("b::x::0", "Second unrelated chunk about notice periods."),
        _chunk("c::x::0", "Third unrelated chunk about hearing dates."),
    ]
    question = "Zorvathi plimquex nendrable?"

    result = rerank(question, chunks, top_k=5)

    scores = {c["rerank_score"] for c in result}
    assert scores == {0}  # a genuine three-way tie
    assert [c["chunk_id"] for c in result] == ["a::x::0", "b::x::0", "c::x::0"]


def test_rerank_empty_input_returns_empty_list():
    assert rerank("anything", [], top_k=5) == []


def test_rerank_never_invents_a_chunk_not_in_the_input():
    chunks = [_chunk("a::x::0", "some content about notice periods")]
    result = rerank("notice period", chunks, top_k=10)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "a::x::0"


def test_rerank_preserves_all_original_fields_and_adds_rerank_score():
    chunk = _chunk("a::x::0", "rent and deposit terms", score=0.71)
    result = rerank("rent and deposit", [chunk], top_k=1)
    assert result[0]["score"] == 0.71
    assert result[0]["section"] == "x"
    assert "rerank_score" in result[0]

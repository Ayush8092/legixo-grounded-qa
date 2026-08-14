"""Offline unit tests for the graph's branch decision (app.graph.route_after_grade).

No network calls, no API keys required — this is a pure function, kept
separate from QAService specifically so the branch/loop logic (assignment
eval criteria #2 and #3) can be tested without building real provider
clients.
"""

from app.graph import route_after_grade


def test_sufficient_grade_routes_to_generate_answer():
    grade = {"sufficient": True, "relevant_chunk_ids": ["a::b::0"], "reason": "ok"}
    assert route_after_grade(grade, loops=0, max_retrieval_loops=2) == "generate_answer"


def test_insufficient_grade_with_loops_remaining_routes_to_rewrite():
    grade = {"sufficient": False, "relevant_chunk_ids": [], "reason": "no match"}
    assert route_after_grade(grade, loops=0, max_retrieval_loops=2) == "rewrite_query"
    assert route_after_grade(grade, loops=1, max_retrieval_loops=2) == "rewrite_query"


def test_insufficient_grade_with_loops_exhausted_routes_to_refuse():
    grade = {"sufficient": False, "relevant_chunk_ids": [], "reason": "no match"}
    assert route_after_grade(grade, loops=2, max_retrieval_loops=2) == "refuse"


def test_loop_bound_is_never_exceeded_regardless_of_how_high_loops_gets():
    """The graph must never loop forever — once loops >= max, always refuse."""
    grade = {"sufficient": False, "relevant_chunk_ids": [], "reason": "no match"}
    for loops in range(2, 50):
        assert route_after_grade(grade, loops=loops, max_retrieval_loops=2) == "refuse"


def test_sufficient_always_wins_even_at_the_loop_boundary():
    grade = {"sufficient": True, "relevant_chunk_ids": ["a::b::0"], "reason": "ok"}
    assert route_after_grade(grade, loops=2, max_retrieval_loops=2) == "generate_answer"


def test_zero_max_retrieval_loops_means_no_retry_at_all():
    grade = {"sufficient": False, "relevant_chunk_ids": [], "reason": "no match"}
    assert route_after_grade(grade, loops=0, max_retrieval_loops=0) == "refuse"

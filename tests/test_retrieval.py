"""Offline unit tests for app.retrieval — score filtering, dedup, pooling.

The Pinecone index and embedder are replaced with small fakes, so these
tests exercise retrieve_single/retrieve_pooled without any network call.
"""

from app.config import Settings
from app.retrieval import retrieve_pooled, retrieve_single


def _settings(**overrides) -> Settings:
    base = dict(
        GEMINI_API_KEY="x",
        GROQ_API_KEY="x",
        PINECONE_API_KEY="x",
        TOP_K=5,
        SCORE_THRESHOLD=0.55,
    )
    base.update(overrides)
    return Settings(**base)


class _FakeEmbeddings:
    """embed_query returns a fixed vector per query text; content doesn't matter
    since the fake index below ignores the vector and just returns canned matches."""

    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]


class _FakeIndex:
    """Returns a canned set of matches per query text, keyed by the query string
    passed through a side channel (the test sets `.responses` directly)."""

    def __init__(self, matches: list[dict]):
        self._matches = matches
        self.query_calls = 0

    def query(self, vector, top_k, namespace, include_metadata):
        self.query_calls += 1
        return {"matches": self._matches}


def _match(chunk_id, score, source_file="doc.md", section="Section", text="text"):
    return {
        "id": chunk_id,
        "score": score,
        "metadata": {
            "chunk_id": chunk_id,
            "source_file": source_file,
            "section": section,
            "text": text,
        },
    }


def test_retrieve_single_filters_by_score_threshold():
    matches = [_match("a::0", 0.9), _match("b::0", 0.3)]
    index = _FakeIndex(matches)
    settings = _settings(SCORE_THRESHOLD=0.55)
    results = retrieve_single(index, _FakeEmbeddings(), settings, "question")
    assert [r["chunk_id"] for r in results] == ["a::0"]


def test_retrieve_single_returns_empty_when_nothing_clears_threshold():
    matches = [_match("a::0", 0.1)]
    index = _FakeIndex(matches)
    settings = _settings(SCORE_THRESHOLD=0.55)
    results = retrieve_single(index, _FakeEmbeddings(), settings, "question")
    assert results == []


def test_retrieve_single_makes_exactly_one_query_call():
    index = _FakeIndex([_match("a::0", 0.9)])
    settings = _settings()
    retrieve_single(index, _FakeEmbeddings(), settings, "question")
    assert index.query_calls == 1


class _MultiQueryFakeIndex:
    """Returns different matches depending on which call number this is —
    used to simulate different queries surfacing different/overlapping chunks."""

    def __init__(self, responses_by_call: list[list[dict]]):
        self._responses = responses_by_call
        self.query_calls = 0

    def query(self, vector, top_k, namespace, include_metadata):
        matches = self._responses[self.query_calls]
        self.query_calls += 1
        return {"matches": matches}


def test_retrieve_pooled_queries_once_per_query_string():
    index = _MultiQueryFakeIndex([[_match("a::0", 0.9)], [_match("b::0", 0.8)]])
    settings = _settings()
    results = retrieve_pooled(index, _FakeEmbeddings(), settings, ["q1", "q2"])
    assert index.query_calls == 2
    assert {r["chunk_id"] for r in results} == {"a::0", "b::0"}


def test_retrieve_pooled_dedupes_and_keeps_best_score():
    index = _MultiQueryFakeIndex([[_match("a::0", 0.6)], [_match("a::0", 0.9)]])
    settings = _settings()
    results = retrieve_pooled(index, _FakeEmbeddings(), settings, ["q1", "q2"])
    assert len(results) == 1
    assert results[0]["score"] == 0.9  # best score across queries wins


def test_retrieve_pooled_sorts_best_first():
    index = _MultiQueryFakeIndex([[_match("a::0", 0.6)], [_match("b::0", 0.9)]])
    settings = _settings()
    results = retrieve_pooled(index, _FakeEmbeddings(), settings, ["q1", "q2"])
    assert [r["chunk_id"] for r in results] == ["b::0", "a::0"]


def test_retrieve_pooled_filters_by_score_threshold_per_query():
    index = _MultiQueryFakeIndex([[_match("a::0", 0.9)], [_match("b::0", 0.1)]])
    settings = _settings(SCORE_THRESHOLD=0.55)
    results = retrieve_pooled(index, _FakeEmbeddings(), settings, ["q1", "q2"])
    assert [r["chunk_id"] for r in results] == ["a::0"]


def test_retrieve_pooled_with_empty_query_list_returns_empty():
    index = _MultiQueryFakeIndex([])
    settings = _settings()
    results = retrieve_pooled(index, _FakeEmbeddings(), settings, [])
    assert results == []
    assert index.query_calls == 0

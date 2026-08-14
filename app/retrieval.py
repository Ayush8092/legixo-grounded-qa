"""Dense retrieval: embed query/queries, search Pinecone, filter, dedupe.

Two entry points:

- `retrieve_single` — one query in, embed once, search once. This is the
  normal path used on the first attempt for every question, so we don't pay
  for multiple embedding + search calls when a single query is enough
  (the corpus is small and free-tier quota matters — see docs/architecture.md).
- `retrieve_pooled` — several queries in (used only on the bounded retry
  after the first attempt is graded insufficient). Each query is searched
  independently and results are pooled and deduplicated by chunk_id, keeping
  the best score a chunk earned across queries.

`SCORE_THRESHOLD` is a retrieval-time recall filter, not proof that a chunk
answers the question — that judgment is left to `llm.grade_chunks`, which is
what actually gates whether we generate an answer or retry (Improvement 4 /
"do not rely exclusively on similarity score for grounding").
"""

from app import vectorstore


def _search(index, embeddings, settings, query: str) -> list[dict]:
    vector = embeddings.embed_query(query)
    return vectorstore.query(index, settings, vector, settings.top_k)


def retrieve_single(index, embeddings, settings, query: str) -> list[dict]:
    """Embed and search with a single query; apply the score floor."""
    matches = _search(index, embeddings, settings, query)
    return [m for m in matches if m["score"] >= settings.score_threshold]


def retrieve_pooled(index, embeddings, settings, queries: list[str]) -> list[dict]:
    """Search with several queries, pool, dedupe by chunk_id, filter by score.

    Keeps the highest score seen for a chunk across all queries, then returns
    results sorted best-first.
    """
    pooled: dict[str, dict] = {}
    for query in queries:
        for match in _search(index, embeddings, settings, query):
            if match["score"] < settings.score_threshold:
                continue
            existing = pooled.get(match["chunk_id"])
            if existing is None or match["score"] > existing["score"]:
                pooled[match["chunk_id"]] = match
    return sorted(pooled.values(), key=lambda m: m["score"], reverse=True)

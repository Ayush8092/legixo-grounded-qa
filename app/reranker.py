"""Lightweight lexical reranking of retrieved chunks.

Pinecone's dense vector search is optimized for recall — it's good at
finding chunks that are broadly about the right topic, but on a small
corpus its ordering can be a poor proxy for "does this chunk actually
answer the question" (cosine scores cluster closely when there are only a
handful of short chunks — see docs/architecture.md, "TOP_K/SCORE_THRESHOLD").
Reranking re-scores Pinecone's candidates against the literal words of the
question before the LLM grader ever sees them, improving ordering without
discarding recall.

Implementation choice: a small, self-contained Okapi BM25 scorer, stdlib
only — no extra dependency, no model download, no external API call. This
is deliberately not a cross-encoder/transformer reranker: for a handful of
short legal-note chunks, a lexical term-overlap reranker is trivial to
install (nothing to install), easy to explain, effectively free to run, and
a good match for a small take-home project (see the reranking requirements
in final_improvement_lexido.docx: "appropriate for a small take-home
project", "do NOT introduce a huge model", "do NOT download several GB of
models"). If the corpus grows much larger or more semantically subtle, a
cross-encoder (e.g. `sentence-transformers` `cross-encoder/ms-marco-*`) would
be the natural next step — see README.md "Reranking" section.

This module ONLY reorders/selects among chunks Pinecone already retrieved.
It never generates text, never invents a chunk, and never talks to an LLM —
grounding, citation validation, and answer generation are entirely
unaffected by its presence (see the `rerank` node in `app/graph.py`).
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Standard Okapi BM25 free parameters (the usual textbook defaults).
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _bm25_scores(query_tokens: list[str], doc_token_lists: list[list[str]]) -> list[float]:
    """Okapi BM25 score of `query_tokens` against each entry in `doc_token_lists`."""
    n_docs = len(doc_token_lists)
    if n_docs == 0:
        return []

    doc_lens = [len(toks) for toks in doc_token_lists]
    avg_len = (sum(doc_lens) / n_docs) if n_docs else 0.0

    doc_freq: Counter[str] = Counter()
    for toks in doc_token_lists:
        for term in set(toks):
            doc_freq[term] += 1

    def idf(term: str) -> float:
        n_qualifying = doc_freq.get(term, 0)
        # Standard BM25 IDF with a +1-style floor so it never goes negative
        # for a term that appears in every single chunk.
        return math.log(1 + (n_docs - n_qualifying + 0.5) / (n_qualifying + 0.5))

    scores: list[float] = []
    for toks, doc_len in zip(doc_token_lists, doc_lens):
        term_freq = Counter(toks)
        score = 0.0
        for term in set(query_tokens):
            freq = term_freq.get(term)
            if not freq:
                continue
            numerator = freq * (_K1 + 1)
            length_norm = 1 - _B + _B * (doc_len / avg_len if avg_len else 1.0)
            denominator = freq + _K1 * length_norm
            score += idf(term) * (numerator / denominator)
        scores.append(score)
    return scores


def rerank(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Reorder `chunks` by lexical relevance to `question`, keep the best `top_k`.

    - Scoring is BM25 over each chunk's `text` — the same context-rich text
      (document title + front matter + section body; see chunking.py) the
      grader and answer step see, so the reranker's notion of "relevant" is
      grounded in exactly what downstream steps will read.
    - Ties — including "BM25 score is 0 for every candidate", which happens
      when the question shares no meaningful terms with any retrieved chunk
      (the out-of-corpus case) — fall back to preserving the input order,
      which `retrieval.py` already returns best-first by vector similarity.
      Python's `sorted` is stable, so this happens automatically. This
      means an out-of-corpus question still hands the grader the same
      (still irrelevant) candidates it always did — reranking doesn't
      invent relevance that isn't there, so abstention is unaffected.
    - `top_k` truncation only ever removes vector-search candidates that
      BM25 also ranks low; it never adds or invents a chunk.
    - Adds a `rerank_score` field to each returned chunk (useful for the
      trace/debugging); every other field from the input chunk is preserved
      unchanged.
    """
    if not chunks:
        return []

    query_tokens = _tokenize(question)
    doc_token_lists = [_tokenize(c.get("text", "")) for c in chunks]
    bm25 = _bm25_scores(query_tokens, doc_token_lists)

    order = sorted(range(len(chunks)), key=lambda i: bm25[i], reverse=True)

    reranked = []
    for i in order[:top_k]:
        item = dict(chunks[i])
        item["rerank_score"] = bm25[i]
        reranked.append(item)
    return reranked

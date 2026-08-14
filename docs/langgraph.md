# LangGraph workflow

`app/graph.py` builds one `StateGraph` per process (inside `QAService`) and
runs every `/ask` request through it. This document is the map: what each
node does, how the one branch works, and how the retry loop is bounded.

## Graph shape

```
retrieve -> rerank -> grade_chunks -[sufficient]-> generate_answer -> validate_citations -> END
                           |
                   [insufficient, loops < MAX_RETRIEVAL_LOOPS] -> rewrite_query -> retrieve  (loops back through rerank)
                           |
                   [insufficient, loops exhausted] -> refuse -> END
```

## Nodes

| Node | Responsibility | Calls out to |
|---|---|---|
| `retrieve` | Embed the current query (or queries, on retry) and search Pinecone. Filters by `SCORE_THRESHOLD`. On the first pass this is a single query (`question`); on retry it is `QUERY_FANOUT` pooled/deduplicated queries. | `app.retrieval`, `app.vectorstore` |
| `rerank` | Reorders/trims Pinecone's candidates by lexical (BM25) relevance to the question, keeping the top `RERANK_TOP_K`. Runs on every pass through the loop (initial attempt and every retry), since it sits directly on the only path into `grade_chunks`. Disableable via `RERANK_ENABLED=false`; a passthrough when disabled or when there are no candidates. Never invents a chunk, never touches citations. | `app.reranker` (no model, no external call) |
| `grade_chunks` | Asks Groq (JSON mode) whether the retrieved chunks are sufficient to answer the question, and which chunk_ids are actually relevant. Returns structured `{sufficient, relevant_chunk_ids, reason}` — **the graph makes the routing decision, not the LLM.** | `app.llm.grade_chunks` |
| `rewrite_query` | Only reached when grading says insufficient and a retry is still allowed. Asks Groq for up to `QUERY_FANOUT` alternative search queries (deduplicated case-insensitively), increments the loop counter. | `app.llm.rewrite_query` |
| `generate_answer` | Passes only the chunks graded relevant to the answer LLM (at `ANSWER_TEMPERATURE`) and asks for a grounded answer plus the chunk_ids it relied on. | `app.llm.generate_answer` |
| `validate_citations` | Re-checks every cited chunk_id in code against the chunks retrieved **for this request**. Any chunk_id that wasn't actually retrieved is dropped silently. If zero citations survive, the node overrides the answer with the refusal text — a model cannot talk its way to an ungrounded answer. | `app.llm.validate_citations` (pure function, no model call) |
| `refuse` | Reached only when the loop budget is exhausted and chunks are still insufficient. Returns the fixed refusal text and no citations. | — |

## The branch (assignment eval criterion #2)

The single required branch is `grade_chunks -> {generate_answer, rewrite_query, refuse}`,
implemented as a plain Python function so it is unit-testable without any
provider client:

```python
# app/graph.py
def route_after_grade(grade: dict, loops: int, max_retrieval_loops: int) -> str:
    if grade.get("sufficient"):
        return "generate_answer"
    if loops < max_retrieval_loops:
        return "rewrite_query"
    return "refuse"
```

See `tests/test_routing.py` for the corresponding tests — including a test
that walks `loops` from 2 to 49 and asserts the route is always `"refuse"`,
i.e. the branch can never accidentally re-enter the retry path once the
budget is spent.

## The loop, and why it's bounded (assignment eval criterion #3)

Two independent mechanisms bound the `rewrite_query -> retrieve -> grade_chunks`
cycle:

1. **`loops` counter vs `MAX_RETRIEVAL_LOOPS`** (default `2`) — the actual,
   intended stop condition, checked in `route_after_grade` above. With the
   default value the graph makes at most one retry: original query, then one
   rewritten attempt, then generate or refuse.
2. **`recursion_limit`** (default `20`) passed to `graph.invoke(...,
   config={"recursion_limit": ...})` — a hard safety net at the LangGraph
   level. If it ever trips (`langgraph.errors.GraphRecursionError`), `main.py`
   catches it and returns a clean refusal response instead of a 500.

The first attempt is deliberately a single query, not a multi-query
fan-out: the corpus is small, and paying for `QUERY_FANOUT` embeddings and
searches on every question (most of which succeed on the first try) would
waste free-tier quota and add latency for no benefit. Fan-out is reserved
for the one retry, when the first attempt is already known to be
insufficient.

`rerank` adds a node to the retry loop but not a new edge back to an
earlier node — `retrieve -> rerank -> grade_chunks` is still a single
forward path, and the only cycle in the graph remains
`rewrite_query -> retrieve`. Adding reranking therefore cannot introduce a
new way for the graph to loop, and both bounding mechanisms above are
unaffected by its presence.

## Where grounding is actually enforced

Two places, both in code rather than only in a prompt:

- `grade_chunks` drops any `relevant_chunk_ids` the grader returned that
  don't correspond to a chunk actually retrieved this request (see
  `app/llm.py::grade_chunks`).
- `validate_citations` re-derives every citation's `source_file`, `section`,
  and `score` from the retrieved chunk's own Pinecone metadata — never from
  anything the answer model said — and drops any `chunk_id` that wasn't
  retrieved this request. If nothing survives, the node forces `found=False`
  and the fixed refusal text, regardless of what `generate_answer` returned.

**Citation determinism (evidence labels).** `generate_answer` never asks
the model to reproduce a `chunk_id` string (e.g.
`01_matter_memo_arvind_v_northfield::next-hearing::0`) — it used to, and
the model would occasionally abbreviate/truncate it, which the exact-match
validator above correctly rejected, silently downgrading an otherwise
fully-grounded answer to "not found". Instead, `app/llm.py::_evidence_context`
labels each candidate chunk `EVIDENCE_1`, `EVIDENCE_2`, ... in the prompt;
the model only ever has to copy back which labels it used
(`evidence_refs`); and `generate_answer` maps each label back to its exact
original `chunk_id` in code before returning `cited_chunk_ids` — the same
key/shape callers (`graph.py`, `validate_citations`) already expected, so
nothing downstream changed. This makes citation *selection* deterministic
and code-controlled without weakening `validate_citations` at all — an
evidence label that wasn't actually shown for this call (hallucinated or
stale) is dropped the same way an invalid `relevant_chunk_ids` entry
already was.

There's also a prompt-level instruction (in `_ANSWER_SYSTEM`) for a specific
partial-information case: if the chunks cover the topic but not the exact
detail asked (e.g. they state a restriction's *duration* but not its
*penalty*), the model is told to say what the documents do state and
explicitly flag the missing detail, rather than inventing it. This is a
prompt-level nudge, not a code-enforced guarantee the way citation
validation is — it's checked in `tests/test_integration.py::test_tempting_but_unsupported_detail_is_not_invented`.

**Entity-name overlap vs. actual topical relevance.** A harder edge case:
"What is the population of Riverside city?" against a corpus that mentions
"Riverside" only as part of a court name and a fictional statute's name.
Lexically this looks close (BM25 will legitimately rank those chunks
relatively high — retrieval's job is recall, and "Riverside" is genuinely a
distinctive shared term), but the chunks say nothing about population; the
match is a coincidental shared proper noun, not a coincidental shared
*topic*. This is handled with defense in depth at two independent points,
not by weakening reranking or the retrieval threshold:
- `_GRADER_SYSTEM` has an explicit worked example for exactly this pattern
  ("a shared entity name or word is not evidence") plus a general rule
  calling out real-world general-knowledge questions (population,
  geography, officeholders, etc.) against an internal legal/contractual
  corpus specifically.
- `_ANSWER_SYSTEM` has a second, independent worked example of the same
  distinction ("missing one sub-detail" vs. "evidence is about something
  else entirely, coincidentally sharing a name") — so even if grading were
  to mis-include an entity-overlap chunk, the answer step has its own,
  separate chance to recognize the evidence doesn't actually address the
  question and still return `found=false`.

Both are prompt-level judgment calls, not code-enforced the way citation
validation is — small/instant-tier models can occasionally disagree with
even an explicit worked example, and provider-side inference at
`temperature=0` is not always bit-for-bit deterministic across calls. What
*is* guaranteed regardless: `validate_citations` still ensures any citation
in the final response was a chunk actually retrieved this request — a
wrong topical judgment cannot escalate into a fabricated citation.

## State

```python
class GraphState(TypedDict):
    question: str                 # original question, never mutated
    queries: list[str]            # current search query/queries
    chunks: list[dict]            # chunks retrieved this request
    grade: dict                   # {sufficient, relevant_chunk_ids, reason}
    loops: int                    # retry counter
    answer: str
    found: bool
    citations: list[dict]         # validated citations only
    trace: Annotated[list[str], operator.add]  # every node appends one line
```

`trace` uses LangGraph's `Annotated[..., operator.add]` reducer so every
node's trace line accumulates rather than overwrites — this is what powers
`POST /ask?trace=true`, and what the frontend's "Behind the scenes" panel
parses to render the docket log, search-query list, and retrieval stats
(see `app/static/app.js`).

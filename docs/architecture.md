# Architecture

## Overview

```
data/corpus/*.md ──ingest──> chunks ──embed (Gemini)──> Pinecone (legixo-corpus)
                                                              |
                                                              v
question ──HTTP /ask──> FastAPI ──> QAService.ask() ──> LangGraph ──> Pinecone (query)
                                                              |
                                                              v
                                                     Groq (grade / rewrite / answer)
                                                              |
                                                              v
                                              {answer, found, citations, trace?}
```

Two independent pipelines share the same settings, chunking, and vectorstore
modules:

- **Ingestion** (`app/ingestion.py`, CLI-only): corpus files → chunks →
  embeddings → Pinecone upsert. Writes to Pinecone; never answers questions.
- **Q&A** (`app/main.py` → `app/graph.py`, HTTP-only): question → LangGraph
  → Pinecone queries + Groq calls → grounded answer. Reads from Pinecone;
  never ingests. There is intentionally no CLI for asking questions.

A static frontend (`app/static/`) is served by the same FastAPI process at
`GET /`, and only ever calls this API's own `/health`, `/ready`, and `/ask`
endpoints — it never talks to Gemini, Groq, or Pinecone directly, and never
sees a provider API key.

## Module responsibilities

| Module | Owns |
|---|---|
| `config.py` | Typed, validated settings loaded from `.env` (`pydantic-settings`). Secrets are `SecretStr`, never a plain `str` — see "Secrets" below. Every other module reads settings through `get_settings()`. |
| `clients.py` | Provider client construction: Gemini embeddings (`Embedder`, via `google-genai`), Groq chat (raw `groq` SDK), Pinecone client/index (create-if-missing for ingestion, fail-fast for the API, dimension-mismatch guard for both). |
| `schemas.py` | FastAPI request/response models (`AskRequest`, `AskResponse`, `Citation`, `HealthResponse`, `ReadyResponse`). |
| `chunking.py` | Pure, offline: Markdown → header-aware, context-rich `Chunk` objects with deterministic IDs and content hashes. No network calls. |
| `vectorstore.py` | Pinecone read/write shape: upsert batching, query normalization, namespace reset, existing-hash lookup, vector count. |
| `ingestion.py` | Orchestrates chunking + embedding + upsert; the `python -m app.ingestion [--reset]` CLI; skips re-embedding chunks whose content hasn't changed. |
| `retrieval.py` | Embed + search + score-filter + dedupe, for both the single-query first attempt and the pooled multi-query retry. |
| `llm.py` | All three LLM-backed steps (`grade_chunks`, `rewrite_query`, `generate_answer`) plus the pure `validate_citations` guard. No LangGraph knowledge. |
| `graph.py` | Wires the above into the `StateGraph`; owns the branch/loop/refuse logic and the `QAService` facade used by the API. |
| `main.py` | FastAPI app: lifespan startup, `GET /`, `GET /health`, `GET /ready`, `POST /ask`, static file mounting, error-type-aware HTTP status mapping. |
| `static/` | The frontend (`index.html`, `styles.css`, `app.js`) — vanilla HTML/CSS/JS, no build step, no framework. |

Each module has exactly one job (Improvement 5 — "separate responsibilities").
Nothing outside `graph.py` knows about LangGraph, and nothing outside
`llm.py` builds a prompt.

## Secrets

`gemini_api_key`, `groq_api_key`, and `pinecone_api_key` are typed as
`pydantic.SecretStr`, not `str`. This means:

- `repr(settings)` and any accidental `print(settings)` show `SecretStr('**********')`,
  never the real value — so a debug log or an exception traceback that
  happens to include the settings object can't leak a key.
- The only place a key is ever unwrapped to plain text is
  `clients.py::get_embeddings` / `get_chat` / `get_pinecone`, via
  `.get_secret_value()`, right at the point it's handed to the provider SDK
  constructor. No other module ever calls `.get_secret_value()`.

## Why the raw provider SDKs (not LangChain wrappers)

Gemini embeddings go through `google-genai` (`app.clients.Embedder`) and
Groq chat goes through the official `groq` SDK, called directly — not
through `langchain-google-genai` / `langchain-groq`. Both SDKs support the
exact features this project actually needs natively:

- `groq`'s `response_format={"type": "json_object"}` makes Groq itself
  guarantee syntactically valid JSON back, which is what `llm.py`'s
  grading/rewriting/answering all depend on — no more regex-extracting
  a JSON object out of a possibly-fenced, possibly-prefixed reply string.
- `groq`'s client has built-in retry/backoff for transient errors
  (`max_retries=`), configured via `GROQ_MAX_RETRIES`.
- `google-genai`'s `embed_content` exposes `task_type`
  (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) and `output_dimensionality`
  directly, which the LangChain wrapper didn't expose as cleanly.

One fewer abstraction layer between this project and the actual provider
behavior it depends on (retry semantics, error types, JSON mode) — and one
fewer package family to keep pinned and compatible. LangGraph is still used
for the workflow orchestration itself; only the provider calls moved off
LangChain's model wrappers.

## Key design decisions

**Why header-aware chunking instead of fixed-size windows?** The corpus is
six short, structurally simple documents (a title, optional front matter,
then `##` sections). Splitting on structure means a chunk is never cut
mid-sentence, and — combined with prefixing the document title and front
matter onto every chunk — a chunk pulled out of context still says what
document and section it's from (Improvements 1 and 2).

**Why is every embedding L2-normalized?** `gemini-embedding-001` only
guarantees unit-length output at its native 3072-dimension output. If
`EMBEDDING_DIMENSION` is set below that (truncating the vector), Google's
own guidance is to renormalize before using cosine similarity — otherwise
Pinecone's cosine metric compares vectors of inconsistent magnitude and
retrieval quality degrades without raising any error. `Embedder._embed`
renormalizes unconditionally, so this is correct regardless of which
dimension is configured.

**Why validate the Pinecone index dimension at startup/ingestion?** This
project has hit exactly the failure this guards against: "Vector dimension
3072 does not match index dimension 768" — a confusing error that used to
only surface deep inside a query or upsert call, well after the process
looked like it had started successfully. `clients._validate_index_dimension`
compares `EMBEDDING_DIMENSION` against the real index's dimension right
after connecting (both in `ensure_index`, used by ingestion, and
`get_index`, used by the API) and raises one clear, actionable error
instead.

**Why single-query retrieval on the first attempt, multi-query on retry
only?** The corpus is six documents; fanning out to `QUERY_FANOUT` embedding
+ search calls on every question (most of which succeed on the first try)
would waste free-tier quota and add latency for no benefit. `QUERY_FANOUT`
is used only on the bounded retry, after the first attempt is already known
to be insufficient.

**Why is similarity score not the grounding signal?** `SCORE_THRESHOLD`
controls what enters the candidate pool, but a high cosine similarity only
means a chunk is topically close — not that it actually answers the
question. `grade_chunks` (an LLM judgment) and `validate_citations` (a pure
code check against what was actually retrieved) are what gate whether an
answer is generated and what citations survive into the response.

**Why does ingestion skip re-embedding unchanged chunks?** Every chunk
carries a deterministic `content_hash` (see `chunking.py`). Before
embedding, `ingestion.run` fetches the `content_hash` metadata already
stored in Pinecone for those chunk IDs (`vectorstore.fetch_existing_hashes`)
and only calls Gemini for chunks whose hash changed or are new. Editing one
section of one document and re-running `python -m app.ingestion` re-embeds
only that section — not the whole six-document corpus. `--reset` disables
this (it wipes the namespace first, so everything is "new" by definition).

**Why does the API fail fast if the Pinecone index doesn't exist, instead of
creating an empty one?** An empty, auto-created index would silently return
zero results forever and look like "every question is unanswerable" rather
than "ingestion hasn't run yet". `clients.get_index` raises a clear
`RuntimeError` that `main.py` turns into a `503` with an actionable message.
Only `app.ingestion` (via `clients.ensure_index`) is allowed to create the
index.

**Why a `lifespan` handler instead of pure lazy initialization?** `main.py`
attempts to build `QAService` once, eagerly, when the process starts — so
the common case (server already running, keys already configured) pays the
client-construction cost once, not per request. If that eager attempt fails
(keys missing, ingestion not run yet, Pinecone briefly unreachable), the
failure is logged and swallowed rather than crashing the server: `/health`
must stay green regardless, and `get_service()` retries lazily on the first
`/ask` call — so starting the server from a fresh `.env`, before running
ingestion, still works once ingestion catches up, without a restart.

**Why a separate `/ready` from `/health`?** A process can be alive
(`/health` green) while genuinely unable to answer a question — keys not
configured, or the corpus not ingested yet. `/health` never touches an
external provider, so it can't tell you that. `/ready` does the same
best-effort service lookup `/ask` would, and reports `ready: false` with a
plain-English `detail` if it fails, without ever raising an HTTP error
itself — useful for a demo reviewer checking "is this actually usable"
before hitting `/ask`, and for container orchestrators distinguishing
liveness from readiness.

## What was deliberately left out

Per Improvement 11 and engineering principle 13, this project does not add:
authentication, a database beyond Pinecone, Redis, a heavy frontend
framework/build step, web search, or a general-purpose agent loop. It is a
bounded, explicit RAG workflow — the LangGraph has exactly seven nodes and
one branch, and the frontend is vanilla HTML/CSS/JS served directly by
FastAPI.

## Reranking

Pinecone's dense-vector search optimizes for recall, not precision — on a
small corpus (a handful of short chunks), cosine scores cluster closely and
the ordering can be a poor proxy for "does this chunk actually answer the
question" (this is why `eval-09`'s multi-document question originally
under-ranked the correct chunk). `app/reranker.py` re-scores Pinecone's
candidates against the literal words of the question using a small,
self-contained Okapi BM25 implementation — stdlib only, **zero new
dependencies**, no model download, no external API call.

This was a deliberate choice over a cross-encoder/transformer reranker
(e.g. `sentence-transformers`): for a handful of short legal-note chunks, a
lexical term-overlap reranker is trivial to install, fast (sub-millisecond),
easy to explain in a review, and appropriately scoped for a small take-home
project. If the corpus grows much larger or the questions get more
semantically subtle (synonyms, paraphrase, no literal term overlap), a
cross-encoder would be the natural next step — the `rerank` node's
interface (`rerank(question, chunks, top_k) -> chunks`) is intentionally
generic enough that swapping the implementation wouldn't require graph
changes.

`rerank` sits as its own node between `retrieve` and `grade_chunks` (see
`docs/langgraph.md`) so it runs on every pass through the retrieval loop —
the initial attempt and every retry after `rewrite_query` — not just the
first one. It only ever reorders/trims chunks Pinecone already retrieved;
it never invents a chunk, never touches citations, and is fully
disableable via `RERANK_ENABLED=false` for debugging/comparison.

## Two knowledge bases, one logical corpus

`data/corpus/` is the immutable, shipped assignment corpus (the six
official documents) — nothing at runtime ever writes into it.
`data/uploads/` is the mutable, runtime knowledge base that `POST /upload`
writes into. Both directories are ingested together, every run, as **one
logical corpus**:

```
data/corpus/  (immutable)  ┐
                            ├─▶  chunk_directories()  ─▶  one chunk list  ─▶  hash/embed/upsert + stale reconciliation
data/uploads/ (mutable)    ┘
```

This split exists so the test suite can keep validating the immutable
baseline (`test_only_the_six_known_source_files_appear`,
`test_chunk_ids_are_deterministic_across_runs`, both in
`tests/test_chunking.py`) independently of whatever a recruiter or
developer has uploaded at runtime — those tests check `data/corpus/`
directly and would (correctly) fail if uploads ever landed there.

**Why not two independent ingestion calls** (`ingest_corpus(CORPUS_DIR)`
then `ingest_corpus(UPLOAD_DIR)`)? Because stale-vector reconciliation
(below) computes `existing_ids - current_chunk_ids`. If each call only knew
about one directory's chunk IDs, the *first* call would see every uploaded
document's vectors as "not part of my current chunk set" and delete them
as stale — and the second call would do the same to the official corpus's
vectors. `app.chunking.chunk_directories()` merges both directories into
one chunk list *before* anything downstream runs, so `current_chunk_ids`
is always the union of both, and neither directory's vectors are ever
mistaken for the other's stale leftovers. See
`app.ingestion.ingest_corpus`'s docstring and
`tests/test_ingestion.py`'s "Prompt_11.docx" test section for the exact
scenarios this protects against.

**Chunk identity across corpus and uploads**: a chunk_id has the shape
`<source_root>::<filename-with-extension>::<section-slug>::<index>`, e.g.
`corpus::02_employment_agreement_excerpt.md::notice-period::0` or
`uploads::point_8_test_documents.txt::leave-policy::0`. Two things this
guards against:
- The **full filename including its extension** (not just the stem) is
  part of the ID, so `notes.txt` and `notes.docx` — or any two files that
  happen to share a stem — never collide even with identical content.
- **`source_root`** (`"corpus"` or `"uploads"`) means a file uploaded with
  the same name as an official document, or the same name as another
  previously-uploaded document from a different logical root, still gets a
  distinct ID rather than silently overwriting the wrong vector.

`source_file` (used for citations and the eval harness's
`expected_source_files` checks) is unaffected — it's still just the plain
filename, e.g. `02_employment_agreement_excerpt.md` — so citations and
`data/corpus/`'s test coverage read exactly as before; only the internal
`chunk_id` used for Pinecone identity changed shape.

## Upload architecture

`POST /upload` (in `app/main.py`) does not implement a second ingestion
pipeline. It:
1. Validates each file (extension, sanitized filename, size limit, not
   empty) before writing anything, with a per-file result — one bad file
   in a batch doesn't fail the others.
2. Writes accepted files into the configured `UPLOAD_DIR` (default
   `data/uploads/`) — never `CORPUS_DIR`. This is what keeps the immutable
   assignment corpus immutable at runtime; see "Two knowledge bases, one
   logical corpus" above.
3. Calls `app.ingestion.ingest_corpus()` — the exact function
   `python -m app.ingestion`'s `main()` also calls, with no directory
   override — so it ingests `CORPUS_DIR` and `UPLOAD_DIR` together, exactly
   like the CLI does. This is also what makes the uploaded document
   immediately retrievable by `/ask` with a citation pointing at its real
   filename: it goes through the identical
   loading/chunking/hashing/embedding/upserting/reconciliation pipeline as
   every other document, not a shortcut.

There is exactly one ingestion pipeline; the CLI and the API are two
callers of the same function, not two implementations.

**Upload security**: filenames are never trusted. `_safe_filename` takes
only the final path component (defeating `../../x` traversal, since a
client-supplied absolute or `../`-prefixed path collapses to its
basename), then replaces anything that isn't alnum/dot/dash/underscore
with `_` (defeating a Windows-style `..\..\x`, which Python's `Path.name`
does not treat as a separator on POSIX), then strips a leading dot (no
hidden/dotfile names). The resulting destination path is re-checked against
the resolved upload directory as defense in depth. Files are read in a
single bounded call (`limit + 1` bytes), so an oversized upload is rejected
without ever being fully loaded into memory. A file that's saved but turns
out corrupt/unreadable (a bad PDF/DOCX) is caught immediately via the same
`app.loaders.load_text()` extraction ingestion itself will run, and removed
— rather than left as a broken file that would only fail later, mid-batch,
for the whole ingestion run.

**Filename collisions**: uploading a file with the same name as an
already-uploaded document overwrites it in `data/uploads/` — this is
treated as "editing that document", consistent with how ingestion already
treats file edits generally: the chunk_id is unchanged (same
`source_root::filename::section::index`), so the content-hash comparison
in `ingest_corpus` naturally re-embeds only the sections that actually
changed, and stale-vector reconciliation cleans up any chunks the previous
version produced that the new version doesn't. Uploading a file with the
*same name as an official corpus document* does **not** overwrite
anything in `data/corpus/` — it's saved into `data/uploads/` instead, and
gets a distinct chunk_id (`uploads::name.md::...` vs `corpus::name.md::...`),
so both remain independently retrievable.

## Stale-vector reconciliation

Deleting a corpus or uploaded file, or removing a section from a file that
survives, previously left old vectors behind in Pinecone forever — a
deleted document could still be retrieved and cited. Every non-`--reset`
ingestion run now also computes:

    stale_ids = <every vector ID currently in the namespace> - <the current corpus + uploads chunk IDs>

and deletes exactly those IDs (`vectorstore.list_all_ids` +
`vectorstore.delete_vectors`) — never `Pinecone`'s `delete_all`, and never
anything outside `settings.pinecone_namespace`. "Current corpus + uploads
chunk IDs" is always computed from **both** directories together (see
"Two knowledge bases, one logical corpus" above) — never from either one in
isolation, which is what makes it safe for uploaded and official vectors to
coexist in the same namespace without either side's reconciliation pass
ever deleting the other's vectors.

**Ownership reasoning**: this project already treats its configured
namespace as exclusively its own — `vectorstore.delete_namespace`, used by
`--reset`, already wipes the *entire* namespace, a strictly more aggressive
operation than "delete only the IDs not in the current corpus". Given that
existing, already-accepted convention, treating every ID found via
`index.list(namespace=...)` as belonging to this project's corpus (and
therefore safe to reconcile) doesn't introduce a new trust boundary; it's
the same one the code already relies on, just used more surgically.

**Graceful degradation**: `index.list(...)` requires a serverless Pinecone

index and a client version that supports it. If unavailable, or if listing
fails for any reason, `list_all_ids` logs a warning and returns an empty
set rather than raising — stale-vector cleanup is skipped for that run, but
the insert/upsert/skip-unchanged behavior that existed before this feature
keeps working exactly as before.

`--reset` skips reconciliation entirely (it already wiped the whole
namespace, so nothing can be stale) rather than performing a redundant
list/delete round-trip.
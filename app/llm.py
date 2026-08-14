"""LLM-backed steps: relevance grading, query rewriting, grounded answering.

The routing decision itself is never made by the LLM (Improvement 4 —
"the LLM provides the judgment, LangGraph makes the actual routing
decision"). Each function here returns plain structured data; `graph.py`
is the only place that decides what to do with it.

Calls go through the raw `groq` SDK (see `app/clients.py`) with
`response_format={"type": "json_object"}`.

Citation validation (`validate_citations`) is pure and has no model calls
at all — it is enforced in code, not by trusting the prompt.
"""

import json
import logging

import groq

_log = logging.getLogger(__name__)


REFUSAL_TEXT = (
    "I cannot find the answer to this question in the provided documents."
)


class LLMProviderError(RuntimeError):
    """A Groq/provider-level failure.

    This represents a provider-level failure such as a rate limit, timeout,
    network failure, or unexpected SDK exception while making the call.

    It is intentionally different from:
    - malformed model output
    - insufficient document evidence

    A provider failure means we do not actually know whether the documents
    answer the question, so it must not be silently converted into an
    "insufficient evidence" result.

    Letting it propagate allows `main.py` to return a 503 instead of
    incorrectly returning a grounded refusal.
    """


_GRADER_SYSTEM = (
    "You are a relevance grader for a document Q&A system. You are given a "
    "question and a numbered list of retrieved candidate chunks, each "
    "labeled CANDIDATE_1, CANDIDATE_2, and so on. Your ONLY job is to "
    "select which candidates, BY LABEL, are relevant evidence for the "
    "question. Judge relevance from each candidate's actual text, never "
    "from its section label or heading alone (a chunk labeled 'Section: "
    "header' can still be exactly on point).\n\n"

    "A chunk is RELEVANT if it is about the same clause, event, or topic "
    "the question asks about — even if it does NOT contain the exact "
    "number or detail the question wants. Missing one specific requested "
    "detail (a penalty, an amount, an extra condition) does NOT make a "
    "chunk irrelevant: it just means that detail is not specified, and the "
    "chunk should still be selected so the answer step can say what IS "
    "stated and flag what is not. Only leave a chunk out if it is not "
    "actually about what the question asks (a different clause, a "
    "different party, a different named term, or an unrelated document).\n\n"

    "A chunk is NOT relevant merely because it contains the same entity "
    "name, a similar word, or belongs to the same general document or "
    "legal matter as the question. Shared wording is not evidence — the "
    "chunk must actually contain information that helps answer what is "
    "being asked. This applies with extra force to real-world general-"
    "knowledge questions (population, geography, current officeholders, "
    "weather, historical events, sports, and similar facts about a real "
    "place, person, or organization named in the corpus) — this corpus is "
    "internal legal/contractual documents, and a location or entity name "
    "appearing in both the question and a chunk is not evidence the chunk "
    "answers a general-knowledge question about that same name; it is "
    "coincidental unless the chunk actually states that specific kind of "
    "fact.\n\n"

    "Critical distinction — a missing SUB-DETAIL of the thing asked about "
    "is not the same as a DIFFERENT, separately-named thing that merely "
    "sits under the same umbrella topic or section heading:\n"
    "- 'Missing sub-detail' (still relevant): the chunk is about the exact "
    "clause, benefit, or event the question asks about, just without one "
    "attribute of it (an amount, a duration, a penalty, a carry-over rule). "
    "Select it.\n"
    "- 'Different named item under a shared umbrella' (NOT relevant, even "
    "though it will often sit right next to, or under the same heading as, "
    "the real answer): the question names a specific category or type "
    "(e.g. 'sick leave' vs 'paid leave'; 'dental insurance' vs 'health "
    "insurance'; 'the confidentiality clause' vs 'the non-compete "
    "clause'), and the chunk instead states a figure or rule for a "
    "*different* specifically-named category. A shared section heading "
    "(e.g. 'Leave and Attendance', 'Benefits') or the general umbrella "
    "word alone does not make the chunk evidence for the specific "
    "category asked about — the chunk must state that specific category, "
    "not just belong to its general family. When in doubt, ask: does this "
    "chunk state a fact about the SAME named thing the question asks "
    "about, or about a sibling item that merely lives near it? Only the "
    "former is relevant.\n\n"

    "Example 1 — Question: 'What penalty applies if X breaches the "
    "non-compete clause?' Chunk: 'The non-compete lasts 12 months after "
    "leaving.' -> RELEVANT. It is the non-compete clause; duration is "
    "stated, penalty is not — select it anyway (missing sub-detail).\n"
    "Example 2 — Question: 'What color is the carpet?' Chunk: 'Rent is "
    "$2,000/month, due on the 1st.' -> NOT relevant. Different topic "
    "entirely, not just missing a detail.\n"
    "Example 3 — a question names a specific term (e.g. a specific act, "
    "code, or clause name): a chunk about a differently named, only "
    "superficially similar term is NOT relevant just because the wording "
    "overlaps — it must be about the specific thing asked about.\n"
    "Example 4 — Question: 'What is the population of Riverside city?' "
    "Chunk: 'the fictional Riverside Code: three years from breach date.' "
    "-> NOT relevant. The word 'Riverside' appears in both, but the chunk "
    "says nothing about population — a shared entity name or word is not "
    "evidence; the chunk must actually contain the fact asked about.\n"
    "Example 5 — Question: 'Who is the president of India?' None of the "
    "chunks shown mention a president or any government official at all. "
    "-> relevant_labels is []. Do not select the least-unrelated chunk "
    "just because the list isn't empty — if nothing shown actually bears "
    "on the question, the correct answer is an empty list, and that is "
    "the expected, correct outcome, not a failure to find something.\n"
    "Example 6 — Question: 'How many sick leave days are employees "
    "entitled to?' Chunk, under a heading like 'Leave and Attendance': "
    "'Employees are entitled to twenty-four (24) paid leave days during "
    "each completed calendar year.' -> NOT relevant. Paid leave and sick "
    "leave are different, specifically-named entitlements; this chunk "
    "states a figure for paid leave only and says nothing about sick "
    "leave. The shared heading/umbrella word 'leave' is not evidence — do "
    "not select this chunk for a sick-leave question (this is the "
    "'different named item under a shared umbrella' case above, not a "
    "missing sub-detail of the same entitlement). If a chunk elsewhere "
    "independently states a sick-leave figure, select that chunk "
    "instead.\n"
    "Example 7 — Question: 'How many sick leave days are employees "
    "entitled to?' Chunk: 'Employees receive 12 sick leave days per "
    "year.' -> RELEVANT. This directly states the sick-leave figure "
    "asked about.\n\n"

    "Each candidate below is ONE indivisible chunk of text, even when it "
    "covers more than one named item (for example a single candidate's "
    "text might mention both health insurance AND dental insurance "
    "figures together, or both paid leave AND sick leave). If a candidate "
    "listed below contains the fact you need, select THAT candidate's "
    "label — never invent or guess at a different, more specific-sounding "
    "label (e.g. a higher number) that was not actually listed, even if it "
    "feels like the 'real' answer should live in its own separate chunk. "
    "Only the labels that literally appear below (CANDIDATE_1 up to the "
    "last one shown) exist; there are no others.\n\n"

    "If the question has multiple parts (e.g. asks for two different "
    "things, or spans more than one document), select every candidate "
    "that is relevant to ANY part — do not stop at the first relevant "
    "one.\n\n"

    "Before finalizing, check each candidate you are about to include: can "
    "you point to a specific sentence in it that actually bears on the "
    "question, not just a shared word, entity name, or document? If the "
    "question names a specific category or type, does that sentence talk "
    "about that SAME category — not a sibling category from the same "
    "family? If not, leave it out — a wrong reason for inclusion (e.g. "
    "'mentions the same city name', or 'mentions leave/benefits in "
    "general') is not a valid reason.\n\n"

    "Reply with a single JSON object and nothing else:\n"
    '{"relevant_labels": ["CANDIDATE_1", ...], "reason": "<one short sentence>"}\n'
    "relevant_labels must only contain labels exactly as shown to you "
    "(e.g. \"CANDIDATE_2\"), never a chunk_id, filename, or anything else "
    "you were not literally given. Include every candidate that meets the "
    "relevance test above — including candidates that only cover part of "
    "the question — and return an empty list whenever nothing shown to "
    "you is actually relevant; an empty list is a normal, correct, and "
    "expected result, not something to avoid."
)


_REWRITE_SYSTEM = (
    "You rewrite a user's question into short, diverse search queries for a "
    "semantic search over a small set of legal-style documents (contracts, "
    "hearing notices, statutes, settlement notes, memos, leases). Use synonyms, "
    "formal/statutory phrasing, and likely party or clause names. Each query is "
    "a short phrase, not a full sentence. Never answer the question, never "
    "invent entities or facts not implied by the question, and keep the "
    "original meaning intact.\n\n"

    "Reply with a single JSON object and nothing else:\n"
    '{"queries": ["<query 1>", "<query 2>", ...]}'
)


_ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered EVIDENCE blocks provided. "
    "Never use outside knowledge and never guess or infer facts that are not "
    "stated in the evidence.\n\n"

    "- Be concise but include the specific figures, dates, and conditions "
    "given in the evidence.\n"

    "- Reference every evidence block you relied on by its label (e.g. "
    "\"EVIDENCE_1\") — never write out a chunk_id or document name "
    "yourself, only the EVIDENCE_N label exactly as given. If evidence "
    "from more than one document is provided, use and reference all of "
    "them that are relevant — do not answer from only one document when "
    "others were also given to you as relevant. You do not have to "
    "reference every evidence block shown to you, only the ones the "
    "answer actually relies on.\n\n"

    "- If the evidence covers the topic but not the specific detail asked "
    "(for example: it states a restriction's duration but not its "
    "penalty), you MUST still answer using what the evidence DOES state, "
    "and explicitly say the missing detail is not specified — never "
    "invent it and never refuse just because one detail is missing. "
    "Example: question 'What penalty applies for breaching the non-compete?', "
    "evidence states only a 12-month duration -> answer something like 'The "
    "agreement states the non-compete lasts 12 months after leaving, under "
    "the stated conditions; it does not specify any penalty for breaching "
    "it.' with found=true and referencing that evidence block — do NOT set "
    "found=false just because the penalty itself isn't in the text.\n\n"

    "- Important distinction: \"the evidence covers the topic but is "
    "missing one sub-detail\" (answer anyway, see above) is different from "
    "\"the evidence happens to mention the same name/place/party as the "
    "question but is actually about something else\" (this is NOT the "
    "topic being asked about — treat it as if no relevant evidence was "
    "given at all). Example: question 'What is the population of "
    "Riverside city?', evidence only states a court name ('District "
    "Court, Riverside Bench') or a limitation period under a law called "
    "the 'Riverside Code' -> found=false, because population is not the "
    "topic of that evidence at all — sharing the word 'Riverside' is not "
    "the same as the evidence being about the question's actual subject. "
    "Contrast this with the non-compete example above, where the evidence "
    "IS about the exact clause asked about, just missing one figure. The "
    "same distinction applies when the question names a specific category "
    "or type and the evidence instead states a figure for a different, "
    "sibling category under the same umbrella (e.g. question asks about "
    "sick leave, evidence states only a paid-leave figure; or question "
    "asks about dental insurance, evidence states only a health-insurance "
    "figure) — treat that as evidence that does NOT address the "
    "question's actual topic, the same as the Riverside case, even though "
    "it was passed to you as 'relevant' evidence for this question.\n\n"

    '- Only set "found" to false — with "answer" set to '
    f'"{REFUSAL_TEXT}" and "evidence_refs" empty — when NONE of the '
    "evidence blocks address the question's actual topic (per the "
    "distinction above).\n\n"

    "Reply with a single JSON object and nothing else:\n"
    '{"found": true|false, "answer": "<answer text>", '
    '"evidence_refs": ["EVIDENCE_1", ...]}\n'
    "evidence_refs must only contain labels exactly as shown to you "
    "(e.g. \"EVIDENCE_2\"), never a chunk_id, filename, or anything else."
)


def _chat_json(
    chat,
    model: str,
    system: str,
    user_text: str,
    temperature: float = 0.0,
) -> dict:
    """Call Groq in JSON mode and return parsed JSON.

    Two failure categories are deliberately handled differently:

    1. A genuine Groq/provider-SDK failure — `groq.APIError` and its
       subclasses (`RateLimitError`, `APITimeoutError`,
       `APIConnectionError`, `InternalServerError`, ...) — means we do not
       actually know whether the documents answer the question. This is
       raised as `LLMProviderError` and MUST propagate out of this
       function (through `grade_chunks`/`rewrite_query`/`generate_answer`,
       through the LangGraph node, to `app/main.py`, which already maps
       `LLMProviderError` to a 503). This is the exact case documented in
       the Groq daily-token-quota `RateLimitError` seen in testing: it must
       surface as "the provider is unavailable", never as a silent
       "insufficient evidence" grade that then burns further quota on a
       pointless `rewrite_query` retry and finally produces a false
       refusal.
    2. Anything else — an unexpected non-provider exception, a response
       shape we don't recognize, an empty reply, or malformed JSON —
       degrades to `{}`, which every caller already treats as "no
       structured data" (insufficient / not found). This is what
       `test_grade_chunks_handles_provider_exception_as_insufficient`
       exercises (with a generic `RuntimeError` standing in for "some
       unexpected bug in the call path", not a real Groq outage), and what
       `test_grade_chunks_handles_malformed_json_as_insufficient` and
       `test_generate_answer_defaults_to_refusal_on_unparseable_reply`
       exercise for bad model output.

    The prints below are the temporary diagnostic logging called for while
    root-causing the live grading failure — they show exactly which branch
    fired instead of guessing. They never print secrets (API keys are never
    passed to this function). Once the live pipeline is confirmed healthy,
    these can be swapped for `_log.debug(...)`.
    """

    try:
        response = chat.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
        )

    except groq.APIError as exc:
        # A real provider/SDK failure (rate limit, timeout, connection,
        # 5xx, ...). Do NOT swallow this into {} — that is what silently
        # turned a Groq 429 into a fake "insufficient evidence" grade.
        print(f"\n[LLM PROVIDER ERROR] model={model} {type(exc).__name__}: {exc}")
        raise LLMProviderError(f"{type(exc).__name__}: {exc}") from exc

    except Exception as exc:
        # An unexpected, non-provider exception (e.g. a bug in a caller,
        # or — in tests — a fake client raising a generic error to
        # simulate "something went wrong"). Not a confirmed provider
        # outage, so this degrades to a safe "no structured data" result
        # rather than raising, matching the existing test contract.
        print(
            f"\n[LLM ERROR] unexpected non-provider exception calling {model}: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}

    try:
        text = response.choices[0].message.content or "{}"
    except (AttributeError, IndexError, TypeError) as exc:
        print(f"\n[LLM ERROR] unexpected response shape from {model}: {type(exc).__name__}: {exc}")
        return {}

    print("\n========== LLM RAW RESPONSE ==========")
    print(f"model={model}")
    print(text)
    print("=======================================\n")

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        # Provider call succeeded, but the model response was malformed.
        print(f"\n[LLM ERROR] malformed JSON from {model}: {exc}")
        return {}

    if not isinstance(data, dict):
        print(f"\n[LLM ERROR] non-object JSON from {model}: {text!r}")
        return {}

    return data


def _candidate_context(chunks: list[dict]) -> tuple[str, dict[str, str]]:
    """Build a CANDIDATE_N-labeled context block for the grading step, plus
    the label -> real chunk_id mapping needed to translate the model's
    reply back (see `grade_chunks`).

    This mirrors `_evidence_context` (used by `generate_answer`) and fixes
    the analogous, observed failure mode in the grading step: shown a
    single retrieved chunk whose text happened to cover more than one
    named sub-topic (e.g. one small chunk mentioning both health insurance
    AND dental insurance figures, because the source document was short
    enough to fit in one chunk), an 8B-class grader model would sometimes
    invent additional chunk_ids that were never actually retrieved — e.g.
    having been shown only `chunk_id: uploads::insurance_test.txt::
    header::0`, it would return `relevant_chunk_ids: ["...header::1",
    "...header::2"]` for the sub-topic it perceived as needing its own
    chunk, extrapolating from the trailing `::0` it saw rather than
    reproducing what was actually shown. The existing `valid_ids` filter
    in `grade_chunks` correctly rejected those fabricated IDs — which is
    the right call, and is NOT weakened here — but the result was the one
    real, relevant, actually-retrieved chunk being filtered out too
    (because it was never included in the model's reply in the first
    place), producing `sufficient=False` for a fully-answerable question.

    Fix (identical in shape to `_evidence_context`'s existing fix for the
    answer step): never ask the model to reproduce or extrapolate a
    chunk_id at all. It only ever sees short, simple labels (`CANDIDATE_1`,
    `CANDIDATE_2`, ...) with no embedded index to extrapolate from, and
    application code — not the model — maps each label back to its exact
    original chunk_id before the `valid_ids` filter runs. The model can no
    longer invent a plausible-looking sibling ID it never had to
    construct in the first place.
    """
    labels: dict[str, str] = {}
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        label = f"CANDIDATE_{i}"
        labels[label] = c["chunk_id"]
        blocks.append(f"{label} [source: {c['source_file']}]\n{c['text']}")
    context = "\n\n".join(blocks) if blocks else "(no chunks retrieved)"
    return context, labels


def _evidence_context(chunks: list[dict]) -> tuple[str, dict[str, str]]:
    """Build an EVIDENCE_N-labeled context block for the answer step, plus
    the label -> real chunk_id mapping needed to translate the model's
    reply back (see `generate_answer`).

    This exists to fix a specific, observed failure mode: asked to echo
    back a `chunk_id` verbatim (e.g.
    `01_matter_memo_arvind_v_northfield::next-hearing::0`), the answer
    model would sometimes truncate/abbreviate it (e.g. drop the
    `::next-hearing::0` suffix). The existing exact-match citation
    validator correctly rejected the malformed ID — which is the right
    call, and is NOT weakened here — but the result was an otherwise
    fully-grounded answer being thrown away and reported as "not found".

    Fix: never ask the model to reproduce a chunk_id at all. It only ever
    sees short, simple labels (`EVIDENCE_1`, `EVIDENCE_2`, ...) it just has
    to copy back, and application code — not the model — maps each label
    to its exact original chunk_id before citation validation runs. The
    model can no longer garble an ID it never had to type in the first
    place; retrieval, grading, and citation validation are all unchanged.
    """
    labels: dict[str, str] = {}
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        label = f"EVIDENCE_{i}"
        labels[label] = c["chunk_id"]
        blocks.append(
            f"{label}\n"
            f"Source: {c['source_file']}\n"
            f"Content:\n{c['text']}"
        )
    context = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    return context, labels


def _dedupe_queries(queries: list[str]) -> list[str]:
    """Strip whitespace and remove blank/duplicate queries.

    Deduplication is case-insensitive while preserving original order.
    """

    seen: set[str] = set()
    out: list[str] = []

    for q in queries:
        q = str(q).strip()
        key = q.lower()

        if q and key not in seen:
            seen.add(key)
            out.append(q)

    return out


def grade_chunks(
    chat,
    model: str,
    question: str,
    chunks: list[dict],
) -> dict:
    """Judge whether retrieved chunks are sufficient to answer the question.

    Returns:

        {
            "sufficient": bool,
            "relevant_chunk_ids": [...],
            "reason": str
        }

    `sufficient` is derived purely in code from whether at least one
    genuinely-retrieved chunk_id survives as relevant — it is NOT taken
    from a separate self-reported "sufficient" field in the model's JSON.
    Asking an 8B-class model to make two overlapping judgments in one call
    (which chunks are relevant, AND whether the whole set is "sufficient")
    is where partial-information questions previously broke: the model
    would correctly identify the on-topic chunk but then separately flag
    "sufficient=false" because one specific requested detail (e.g. a
    penalty) wasn't in it, silently discarding a chunk it had just judged
    relevant. Deriving sufficiency structurally from the relevance list
    removes that redundant, error-prone self-assessment.

    Empty retrieval results are always considered insufficient.

    Malformed/unparseable model output also resolves to insufficient.

    The model never sees or returns a `chunk_id` here either — same as
    `generate_answer` — it sees short `CANDIDATE_N` labels and returns
    `relevant_labels` (see `_candidate_context`'s docstring for the exact
    observed failure this fixes: a model shown one chunk_id would
    sometimes extrapolate sibling IDs like `...header::1`/`...header::2`
    that were never retrieved, rather than reproducing what was shown).
    Application code maps labels back to the exact original chunk_ids
    before the `valid_ids` membership check runs, so that check is still
    the real, unweakened safety net — it now simply can't be defeated by
    the model inventing a plausible-looking ID, because the model never
    constructs an ID string at all.
    """

    if not chunks:
        return {
            "sufficient": False,
            "relevant_chunk_ids": [],
            "reason": "no chunks retrieved",
        }

    context, label_to_chunk_id = _candidate_context(chunks)
    user_text = (
        f"Question: {question}\n\n"
        f"Retrieved candidates:\n"
        f"{context}"
    )

    data = _chat_json(
        chat,
        model,
        _GRADER_SYSTEM,
        user_text,
    )

    valid_ids = {
        c["chunk_id"]
        for c in chunks
    }

    raw_relevant = data.get("relevant_labels", [])

    if not isinstance(raw_relevant, list):
        raw_relevant = []

    # Translate each label back to its real chunk_id first (dropping any
    # label that wasn't actually shown for this call — hallucinated,
    # mistyped, or a stale label from elsewhere); then the same
    # `valid_ids` membership check as before runs on real chunk_ids,
    # exactly as it always has. Belt-and-braces: even if a caller's fake
    # test double still returns raw chunk_ids directly (as the older
    # contract did), a value that happens to already be a valid chunk_id
    # is still accepted — only genuinely unrecognized values are dropped.
    relevant = [
        cid
        for cid in (
            label_to_chunk_id.get(str(ref), str(ref)) for ref in raw_relevant
        )
        if cid in valid_ids
    ]

    return {
        "sufficient": bool(relevant),
        "relevant_chunk_ids": relevant,
        "reason": str(
            data.get("reason", "")
        ).strip(),
    }


def rewrite_query(
    chat,
    model: str,
    question: str,
    previous_query: str,
    fanout: int,
) -> list[str]:
    """Generate up to `fanout` diverse search queries for a bounded retry.

    If query generation fails or returns malformed/empty data, the original
    question is used as a fallback.
    """

    user_text = (
        f"Generate up to {fanout} search queries for this question. "
        "The previous search query found insufficient results, so make "
        "these queries more diverse and go broader where useful.\n\n"
        f"Question: {question}\n"
        f"Previous query: {previous_query}"
    )

    data = _chat_json(
        chat,
        model,
        _REWRITE_SYSTEM,
        user_text,
    )

    raw_queries = data.get("queries", [])

    if not isinstance(raw_queries, list):
        raw_queries = []

    queries = _dedupe_queries(raw_queries)

    if not queries:
        queries = [question]

    return queries[:fanout]


def generate_answer(
    chat,
    model: str,
    question: str,
    chunks: list[dict],
    temperature: float = 0.0,
) -> dict:
    """Generate a grounded answer from retrieved chunks.

    Returns:

        {
            "found": bool,
            "answer": str,
            "cited_chunk_ids": [...]
        }

    The model itself never sees or returns a `chunk_id` — it sees short
    `EVIDENCE_N` labels and returns `evidence_refs` (which labels it used);
    application code maps those back to the exact original chunk_ids via
    the mapping `_evidence_context` built for this exact call, before
    returning them under the same `cited_chunk_ids` key callers already
    expect. This is what makes citation selection deterministic and
    code-controlled rather than dependent on the model reproducing an
    opaque ID string correctly (see `_evidence_context`'s docstring).

    An `evidence_refs` entry that doesn't match a label actually shown for
    this call (hallucinated, mistyped, or referencing a stale label from
    earlier in the conversation) is dropped here — the same
    defensive-filtering pattern `grade_chunks` already uses for
    `relevant_chunk_ids`. Nothing downstream changes: `cited_chunk_ids` is
    still validated separately, unconditionally, by `validate_citations`
    against the chunks actually retrieved this request — this function
    cannot cause a fabricated chunk_id to reach that check, only cause a
    legitimate one to be reported correctly.
    """

    evidence_text, label_to_chunk_id = _evidence_context(chunks)

    user_text = f"Question: {question}\n\nEvidence:\n{evidence_text}"

    data = _chat_json(
        chat,
        model,
        _ANSWER_SYSTEM,
        user_text,
        temperature=temperature,
    )

    found = bool(data.get("found"))

    raw_refs = data.get("evidence_refs", [])
    if not isinstance(raw_refs, list):
        raw_refs = []

    cited_chunk_ids = [
        label_to_chunk_id[str(ref)]
        for ref in raw_refs
        if str(ref) in label_to_chunk_id
    ]

    return {
        "found": found,
        "answer": (
            str(data.get("answer", "")).strip()
            or REFUSAL_TEXT
        ),
        "cited_chunk_ids": cited_chunk_ids,
    }


def validate_citations(
    cited_chunk_ids: list[str],
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Validate citations without making any model calls.

    Level 1:
        The chunk_id must exist among the chunks retrieved for THIS request.

    Level 2:
        Citation metadata is constructed directly from the retrieved chunk,
        never from model output.

    Therefore a fabricated chunk_id, source_file, section, or score cannot
    leak into the API response.
    """

    by_id = {
        c["chunk_id"]: c
        for c in retrieved_chunks
    }

    validated = []
    seen = set()

    for cid in cited_chunk_ids:
        chunk = by_id.get(cid)

        if chunk is None or cid in seen:
            continue

        seen.add(cid)

        validated.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "section": chunk["section"],
                "snippet": chunk["text"][:280],
                "score": round(chunk["score"], 4),
            }
        )

    return validated
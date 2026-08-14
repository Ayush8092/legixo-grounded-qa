"""Evaluation harness: runs eval/test_cases.json against the live graph.

Requires real GEMINI_API_KEY, GROQ_API_KEY, and PINECONE_API_KEY (via .env)
and a Pinecone index already populated by `python -m app.ingestion`.

Usage:
    python -m eval.run_eval

Scores three things per question, per docs/architecture.md section 21:
    1. answer correctness   — every `expected_keywords` entry appears in the
                               answer (case-insensitive); if the case sets
                               `any_of_keywords`, at least one of those must
                               also appear (used for "state this concept, in
                               any of these equivalent phrasings" checks);
                               if the case sets `forbidden_keywords`, none of
                               those may appear (used to catch a specific,
                               plausible-looking wrong answer — e.g. a
                               partial-information question answered with an
                               invented number instead of "not specified").
    2. citation correctness — every citation's source_file is one of
                               `allowed_source_files` (falls back to the
                               older `expected_source_files` key for
                               backward compatibility), AND, if the case
                               sets `required_source_files`, every one of
                               those is actually present in the citations.
                               `allowed_source_files` is a ceiling ("must not
                               cite outside this set"); `required_source_files`
                               is a floor ("must cite at least this") — a
                               case can set the former without the latter, so
                               a question fully answerable from one of two
                               allowed documents still passes with a single
                               citation, unless a case explicitly opts into
                               requiring more.
    3. abstention correctness — `found` matches the `answerable` label

Writes a human-readable report to eval/results.md and exits non-zero if any
question fails, so it can be wired into CI.
"""

import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = EVAL_DIR / "test_cases.json"
RESULTS_PATH = EVAL_DIR / "results.md"


def _keyword_hit(answer: str, keyword: str) -> bool:
    return keyword.lower() in answer.lower()


def _all_keywords_present(answer: str, keywords: list[str]) -> bool:
    return all(_keyword_hit(answer, kw) for kw in keywords)


def _any_keyword_present(answer: str, keywords: list[str]) -> bool:
    """True if `keywords` is empty (nothing required) or at least one hits."""
    if not keywords:
        return True
    return any(_keyword_hit(answer, kw) for kw in keywords)


def _no_forbidden_keyword_present(answer: str, keywords: list[str]) -> bool:
    return not any(_keyword_hit(answer, kw) for kw in keywords)


def score_case(case: dict, result: dict) -> dict:
    abstention_ok = result["found"] == case["answerable"]

    if case["answerable"]:
        answer_ok = (
            _all_keywords_present(result["answer"], case.get("expected_keywords", []))
            and _any_keyword_present(result["answer"], case.get("any_of_keywords", []))
            and _no_forbidden_keyword_present(result["answer"], case.get("forbidden_keywords", []))
        )

        cited_files = {c["source_file"] for c in result["citations"]}
        allowed_files = set(
            case.get("allowed_source_files", case.get("expected_source_files", []))
        )
        required_files = set(case.get("required_source_files", []))
        citation_ok = (
            bool(cited_files)
            and cited_files.issubset(allowed_files)
            and required_files.issubset(cited_files)
        )
    else:
        # An abstention has no keywords to check and must carry no citations.
        answer_ok = not result["found"]
        citation_ok = result["citations"] == []

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "answerable_expected": case["answerable"],
        "found_actual": result["found"],
        "abstention_ok": abstention_ok,
        "answer_ok": answer_ok,
        "citation_ok": citation_ok,
        "passed": abstention_ok and answer_ok and citation_ok,
        "answer": result["answer"],
        "citations": result["citations"],
    }


def run() -> list[dict]:
    from app.graph import QAService
    from app.llm import LLMProviderError

    cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    service = QAService()

    scored = []
    for case in cases:
        started = time.monotonic()
        try:
            result = service.ask(case["question"])
        except LLMProviderError as exc:
            # A real Groq/provider outage (e.g. the daily token-quota
            # RateLimitError) partway through the 16 questions. Record it
            # as its own row and keep going instead of aborting the whole
            # run — a provider outage on question 9 shouldn't cost us the
            # results for questions 1-8 and 10-16, and it must not be
            # scored as if the corpus lacked the answer.
            elapsed = time.monotonic() - started
            row = {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "answerable_expected": case["answerable"],
                "found_actual": None,
                "abstention_ok": False,
                "answer_ok": False,
                "citation_ok": False,
                "passed": False,
                "answer": None,
                "citations": [],
                "trace": [],
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(elapsed, 2),
            }
            scored.append(row)
            print(f"[ERROR] {case['id']}: provider failure — {row['error']}")
            continue

        elapsed = time.monotonic() - started
        row = score_case(case, result)
        row["trace"] = result.get("trace", [])
        row["error"] = None
        row["seconds"] = round(elapsed, 2)
        scored.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status}] {case['id']}: {case['question']}")
    return scored


def write_report(scored: list[dict]) -> None:
    total = len(scored)
    passed = sum(1 for r in scored if r["passed"])
    errored = sum(1 for r in scored if r.get("error"))
    abstention_ok = sum(1 for r in scored if r["abstention_ok"])
    answer_ok = sum(1 for r in scored if r["answer_ok"])
    citation_ok = sum(1 for r in scored if r["citation_ok"])

    lines = [
        "# Evaluation results",
        "",
        f"- Total questions: {total}",
        f"- Passed (all three checks): {passed}/{total}",
        f"- Provider errors (not scored as fail/pass — provider was unavailable): {errored}/{total}",
        f"- Abstention correctness: {abstention_ok}/{total}",
        f"- Answer correctness: {answer_ok}/{total}",
        f"- Citation correctness: {citation_ok}/{total}",
        "",
        "| ID | Category | Answerable? | Found? | Pass | Question |",
        "|---|---|---|---|---|---|",
    ]
    for r in scored:
        status = "⚠️ ERROR" if r.get("error") else ("✅" if r["passed"] else "❌")
        lines.append(
            f"| {r['id']} | {r['category']} | {r['answerable_expected']} | "
            f"{r['found_actual']} | {status} | {r['question']} |"
        )

    lines += ["", "## Details", ""]
    for r in scored:
        if r.get("error"):
            lines.append(f"### {r['id']} — PROVIDER ERROR ({r['seconds']}s)")
            lines.append(f"- Question: {r['question']}")
            lines.append(f"- Error: {r['error']}")
            lines.append(
                "- Not scored as pass/fail: the provider (Groq) was unavailable "
                "for this call, so this is not evidence the corpus/grader lacks "
                "the answer."
            )
            lines.append("")
            continue

        lines.append(f"### {r['id']} — {'PASS' if r['passed'] else 'FAIL'} ({r['seconds']}s)")
        lines.append(f"- Question: {r['question']}")
        lines.append(f"- Answer: {r['answer']}")
        lines.append(
            f"- Citations: {[c['chunk_id'] for c in r['citations']] or '(none)'}"
        )
        lines.append(
            f"- Checks: abstention={r['abstention_ok']} answer={r['answer_ok']} "
            f"citation={r['citation_ok']}"
        )
        if r.get("trace"):
            lines.append("- Trace:")
            for line in r["trace"]:
                lines.append(f"  - {line}")
        lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print(f"{passed}/{total} passed, {errored}/{total} provider errors")


def main() -> None:
    scored = run()
    write_report(scored)
    # Provider errors are excluded from the pass/fail exit code — they mean
    # "couldn't test this question right now" (e.g. Groq daily quota),
    # not "the application got it wrong". A genuine FAIL still exits non-zero.
    genuine_failures = [r for r in scored if not r["passed"] and not r.get("error")]
    if genuine_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

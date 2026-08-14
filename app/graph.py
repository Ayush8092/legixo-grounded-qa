"""LangGraph workflow for grounded Q&A over the corpus.

Graph shape (matches docs/langgraph.md):

    START -> retrieve -> rerank -> grade_chunks --(sufficient)--> generate_answer -> validate_citations -> END
                                        |
                                 (insufficient, loops < MAX_RETRIEVAL_LOOPS)
                                        v
                                   rewrite_query -> retrieve  (loops back into rerank -> grade_chunks)
                                        |
                                 (loops exhausted)
                                        v
                                      refuse -> END

`rerank` sits between `retrieve` and `grade_chunks` as its own explicit
node — not folded into either — so the graph stays readable and so it runs
on *every* pass through the retrieve loop (the initial attempt and every
retry after `rewrite_query`), not just the first one: `rewrite_query`
always routes back to `retrieve`, and `retrieve` always routes to `rerank`,
so there is exactly one path into grading and it always goes through
reranking. See `app/reranker.py` for what it does and why (lightweight,
lexical, no model download, disableable via `RERANK_ENABLED=false`).

Guardrails encoded here (Improvement 8 / engineering principles 6-10):
- `grade_chunks` is the one required branch node — the LLM only supplies a
  judgment; `_route_after_grade` is a plain Python function that makes the
  actual routing decision.
- The `loops` counter plus `recursion_limit` passed to `graph.invoke` both
  bound the retry loop, so the graph can never run forever even if the
  counter logic had a bug. Reranking adds a node to the loop but not a new
  edge back to an earlier node, so it cannot introduce a new cycle.
- `validate_citations` re-checks every cited chunk_id against the chunks
  actually retrieved for this request, so a fabricated citation can never
  reach the API response, regardless of what the answer model claims.
  Reranking only reorders/trims candidates *before* grading — it never
  touches citations or the answer, so this guarantee is unaffected by it.
- The first retrieval attempt is always a single query (cheap); the
  bounded retry is the only place multi-query fan-out (QUERY_FANOUT) is used.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from app import llm, reranker, retrieval
from app.clients import get_chat, get_embeddings, get_index, get_pinecone
from app.config import Settings, get_settings
from app.llm import REFUSAL_TEXT


class GraphState(TypedDict):
    question: str
    queries: list[str]
    chunks: list[dict]
    grade: dict
    loops: int
    answer: str
    found: bool
    citations: list[str] | list[dict]
    trace: Annotated[list[str], operator.add]




def route_after_grade(grade: dict, loops: int, max_retrieval_loops: int) -> str:
    """Pure branch decision — factored out of the class so it is unit-testable
    without building real Gemini/Groq/Pinecone clients.

    This is the one required branch point in the graph (assignment eval
    criterion #2): "sufficient" takes the good path, otherwise we retry up
    to `max_retrieval_loops` times before refusing.
    """
    if grade.get("sufficient"):
        return "generate_answer"
    if loops < max_retrieval_loops:
        return "rewrite_query"
    return "refuse"


class QAService:
    """Builds the graph once (per process) and answers questions through it."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chat = get_chat(self.settings)
        self.embeddings = get_embeddings(self.settings)
        self.pc = get_pinecone(self.settings)
        self.index = get_index(self.pc, self.settings)
        self.graph = self._build_graph()

    # ---- nodes --------------------------------------------------------

    def _retrieve(self, state: GraphState) -> dict:
        if state.get("loops", 0) == 0:
            chunks = retrieval.retrieve_single(
                self.index, self.embeddings, self.settings, state["question"]
            )
            queries = [state["question"]]
        else:
            queries = state["queries"]
            chunks = retrieval.retrieve_pooled(
                self.index, self.embeddings, self.settings, queries
            )
        return {
            "chunks": chunks,
            "trace": [
                f"retrieve: queries={queries} -> {len(chunks)} chunks above "
                f"score>={self.settings.score_threshold} "
                f"{[c['chunk_id'] for c in chunks]}"
            ],
        }

    def _rerank(self, state: GraphState) -> dict:
        """Reorder/trim `state['chunks']` by lexical relevance before grading.

        Runs on every pass through the retrieve loop (see module docstring).
        Disableable via `RERANK_ENABLED=false` for debugging/comparison —
        when disabled, or when there's nothing to rerank, this is a
        passthrough and the original retrieve -> grade flow is unchanged.
        """
        chunks = state["chunks"]
        if not self.settings.rerank_enabled or not chunks:
            return {
                "trace": [
                    f"rerank: skipped (enabled={self.settings.rerank_enabled}, "
                    f"{len(chunks)} candidates)"
                ]
            }
        reranked = reranker.rerank(state["question"], chunks, self.settings.rerank_top_k)
        return {
            "chunks": reranked,
            "trace": [
                f"rerank: {len(chunks)} candidates -> top {len(reranked)} "
                f"{[c['chunk_id'] for c in reranked]}"
            ],
        }

    def _grade_chunks(self, state: GraphState) -> dict:
        grade = llm.grade_chunks(self.chat, self.settings.answer_model, state["question"], state["chunks"])
        return {
            "grade": grade,
            "trace": [
                f"grade_chunks: sufficient={grade['sufficient']} "
                f"relevant={grade['relevant_chunk_ids']} reason={grade['reason']!r}"
            ],
        }

    def _rewrite_query(self, state: GraphState) -> dict:
        loops = state.get("loops", 0) + 1
        previous = state["queries"][0] if state.get("queries") else state["question"]
        queries = llm.rewrite_query(
            self.chat, self.settings.answer_model, state["question"], previous, self.settings.query_fanout
        )
        return {
            "queries": queries,
            "loops": loops,
            "trace": [f"rewrite_query (loop {loops}/{self.settings.max_retrieval_loops}): {queries}"],
        }

    def _generate_answer(self, state: GraphState) -> dict:
        relevant_ids = set(state["grade"]["relevant_chunk_ids"])
        relevant_chunks = [c for c in state["chunks"] if c["chunk_id"] in relevant_ids]
        result = llm.generate_answer(
            self.chat,
            self.settings.answer_model,
            state["question"],
            relevant_chunks,
            temperature=self.settings.answer_temperature,
        )
        return {
            "found": result["found"],
            "answer": result["answer"],
            "citations": result["cited_chunk_ids"],  # raw IDs; validated next
            "trace": [
                f"generate_answer: found={result['found']} cited={result['cited_chunk_ids']}"
            ],
        }

    def _validate_citations(self, state: GraphState) -> dict:
        cited_ids = state["citations"]
        valid = llm.validate_citations(cited_ids, state["chunks"])
        dropped = len(set(cited_ids)) - len(valid)
        if not state.get("found") or not valid:
            return {
                "found": False,
                "answer": REFUSAL_TEXT,
                "citations": [],
                "trace": [
                    f"validate_citations: {len(valid)} valid, {dropped} dropped -> refusal "
                    "(no grounded citation survived validation)"
                ],
            }
        return {
            "citations": valid,
            "trace": [f"validate_citations: {len(valid)} valid, {dropped} dropped"],
        }

    def _refuse(self, state: GraphState) -> dict:
        return {
            "found": False,
            "answer": REFUSAL_TEXT,
            "citations": [],
            "trace": [
                f"refuse: retrieval loops exhausted "
                f"({state.get('loops', 0)}/{self.settings.max_retrieval_loops})"
            ],
        }

    # ---- routing --------------------------------------------------------

    def _route_after_grade(self, state: GraphState) -> str:
        return route_after_grade(
            state["grade"], state.get("loops", 0), self.settings.max_retrieval_loops
        )

    # ---- graph ------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("grade_chunks", self._grade_chunks)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_node("refuse", self._refuse)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "grade_chunks")
        graph.add_conditional_edges(
            "grade_chunks",
            self._route_after_grade,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "refuse": "refuse",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate_answer", "validate_citations")
        graph.add_edge("validate_citations", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    # ---- public API ---------------------------------------------------

    def ask(self, question: str) -> dict:
        initial: GraphState = {
            "question": question,
            "queries": [],
            "chunks": [],
            "grade": {},
            "loops": 0,
            "answer": "",
            "found": False,
            "citations": [],
            "trace": [],
        }
        final = self.graph.invoke(
            initial, config={"recursion_limit": self.settings.recursion_limit}
        )
        return {
            "answer": final["answer"],
            "found": final["found"],
            "citations": final["citations"],
            "trace": final["trace"],
        }

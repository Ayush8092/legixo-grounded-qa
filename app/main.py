"""FastAPI app exposing the Q&A graph over HTTP, plus the static frontend.

Endpoints:
GET  /                 - the frontend (app/static/index.html)
GET  /health            - liveness: process is up, no external calls
GET  /ready             - readiness: settings + Pinecone index are usable
POST /ask              - {"question": "..."} -> {"answer", "found", "citations", "trace"?}
POST /ask?trace=true   - same, but include the LangGraph execution trace
POST /upload            - multipart file upload (.md/.txt/.pdf/.docx) -> saved to
                           data/uploads/ (never data/corpus/) and ingested via the
                           same pipeline as `python -m app.ingestion`

Q&A is only ever reachable through this HTTP API — there is intentionally no
CLI for asking questions (ingestion is the only CLI entry point, per the
assignment brief). `POST /upload` is additive: it does not change `/ask`'s
request or response shape, and it calls `app.ingestion.ingest_corpus`
directly rather than re-implementing any ingestion logic. Uploaded files
never touch `data/corpus/` (the immutable, shipped assignment corpus) — see
docs/architecture.md, "Two knowledge bases, one logical corpus".

Startup architecture: `lifespan` makes a best-effort attempt to build
`QAService` once, eagerly, when the process starts, so the common case is
zero per-request client construction. If that fails (missing keys, index not
ingested yet, Pinecone briefly unreachable), the failure is logged and
swallowed rather than crashing the whole app — `/health` must stay up
regardless, and `get_service()` retries lazily on the first `/ask` call, so
the server started from a clean `.env` before running ingestion still works
once ingestion catches up.
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from pathlib import Path

import groq
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
from langgraph.errors import GraphRecursionError
from pinecone.exceptions import PineconeException
from pydantic import ValidationError

from app.config import get_settings
from app.graph import QAService
from app.ingestion import _resolve_upload_dir, ingest_corpus
from app.llm import LLMProviderError
from app.loaders import LoaderError, SUPPORTED_EXTENSIONS, load_text
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadyResponse,
    UploadRejection,
    UploadResponse,
)


_log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_UPSTREAM_ERRORS = (
    LLMProviderError,
    groq.APIError,
    genai_errors.APIError,
    PineconeException,
)

_service: QAService | None = None

_init_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="qaservice-init",
)


def get_service() -> QAService:
    """Lazy singleton fallback — used if eager startup init in `lifespan` failed.

    Bounded by `SERVICE_INIT_TIMEOUT_SECONDS`: if a provider is unreachable
    (bad network, wrong region, DNS issue), this raises a clear `RuntimeError`
    instead of hanging the request forever. `/ready` and `/ask` both go
    through this, so both get the same bounded, fail-fast behavior.
    """
    global _service

    if _service is None:
        settings = get_settings()

        future = _init_executor.submit(QAService)

        try:
            _service = future.result(
                timeout=settings.service_init_timeout_seconds
            )
        except FutureTimeoutError as exc:
            raise RuntimeError(
                f"Timed out after {settings.service_init_timeout_seconds}s "
                "initializing Gemini/Groq/Pinecone clients. Check network "
                "connectivity and API keys "
                "(SERVICE_INIT_TIMEOUT_SECONDS in .env controls this)."
            ) from exc

    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Kick off QAService construction in the background; never block startup.

    `QAService()` makes synchronous network calls (Pinecone, and potentially
    Gemini/Groq client setup). Awaiting it directly here would block the
    event loop until it finishes — and if a provider is slow or unreachable
    (bad network, DNS issue, wrong region), that can hang for the SDK's full
    connect timeout.

    Running it in a background thread via `run_in_executor` means `yield`
    happens immediately regardless of how long (or whether) that call
    finishes.
    """

    async def _warm_up() -> None:
        global _service

        try:
            loop = asyncio.get_running_loop()

            _service = await loop.run_in_executor(
                None,
                QAService,
            )

            _log.info("QAService initialized at startup.")

        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "QAService could not be initialized at startup (%s: %s). "
                "The server is still up; /health stays up, and /ask will "
                "retry initialization on first use. Common causes: missing "
                "API keys in .env, or `python -m app.ingestion` hasn't been "
                "run yet.",
                type(exc).__name__,
                exc,
            )

    asyncio.create_task(_warm_up())

    yield


app = FastAPI(
    title="Legixo Grounded Q&A API",
    description=(
        "Document-grounded Q&A over a fictional legal corpus. "
        "Answers are restricted to retrieved Pinecone chunks; every answer "
        "carries citations, and unsupported questions are explicitly refused."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


if _STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=_STATIC_DIR),
        name="static",
    )


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the chat UI. It only ever talks to this API's own /ask endpoint."""

    index_file = _STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not built (app/static/index.html missing).",
        )

    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only: the process is up.

    Never touches Pinecone/Gemini/Groq.
    """
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness: can this process actually answer a question right now?

    Distinct from /health on purpose: a container can be alive
    (process running, /health green) while not yet ready
    (ingestion hasn't run, or keys are misconfigured).
    """

    settings = get_settings()

    try:
        service = get_service()

    except ValidationError as exc:
        return ReadyResponse(
            ready=False,
            detail=f"Configuration error: {_short(exc)}",
        )

    except RuntimeError as exc:
        return ReadyResponse(
            ready=False,
            detail=str(exc),
        )

    except Exception as exc:  # noqa: BLE001
        return ReadyResponse(
            ready=False,
            detail=f"{type(exc).__name__}: {_short(exc)}",
        )

    return ReadyResponse(
        ready=True,
        detail=(
            f"index={settings.pinecone_index_name} "
            f"namespace={settings.pinecone_namespace} "
            f"model={settings.answer_model}"
        ),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
)
def ask(
    request: AskRequest,
    trace: bool = Query(
        default=False,
        description="Include the LangGraph execution trace in the response.",
    ),
) -> AskResponse:

    settings = get_settings()

    try:
        service = get_service()

        result = service.ask(request.question)

    except ValidationError as exc:
        # Missing/invalid settings.
        raise HTTPException(
            status_code=500,
            detail=f"Configuration error: {_short(exc)}",
        ) from exc

    except GraphRecursionError as exc:
        # The retry loop's safety-net limit tripped.
        _log.warning("graph recursion limit hit: %s", exc)

        return AskResponse(
            answer=_RECURSION_REFUSAL,
            found=False,
            citations=[],
        )

    except _UPSTREAM_ERRORS as exc:
        # Provider/network failure.
        _log.warning(
            "upstream provider error on /ask: %s: %s",
            type(exc).__name__,
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=f"Upstream service unavailable ({type(exc).__name__}).",
        ) from exc

    except RuntimeError as exc:
        # For example: Pinecone index missing.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:  # noqa: BLE001
        # Genuine unexpected application bug.
        _log.exception("unexpected error on /ask")

        raise HTTPException(
            status_code=500,
            detail="Internal application error.",
        ) from exc

    include_trace = trace or settings.include_trace

    return AskResponse(
        answer=result["answer"],
        found=result["found"],
        citations=result["citations"],
        trace=result["trace"] if include_trace else None,
    )


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(original: str | None) -> str:
    """Turn an untrusted upload filename into a safe, flat filename.

    - `Path(...).name` keeps only the final path component, discarding any
      `../`, a leading `/`, or a drive/absolute prefix a client could send
      — this is the actual path-traversal defense (e.g. `../../.env` ->
      `.env`; an absolute path collapses to its basename).
    - Anything left that isn't alnum/dot/dash/underscore (including a
      literal backslash from a Windows-style `..\\..\\something`, which
      `Path.name` does not treat as a separator on POSIX) is replaced with
      `_`, and a leading dot is stripped — so no hidden/dotfile name and no
      residual `..`-looking prefix can survive.
    `_upload_documents` additionally re-checks the resulting destination
    path is still inside `upload_dir` before writing, as defense in depth.
    """
    name = Path((original or "").strip()).name
    if not name or name in {".", ".."}:
        raise ValueError("missing or invalid filename")
    safe = _SAFE_FILENAME_RE.sub("_", name).lstrip(".")
    if not safe:
        raise ValueError("missing or invalid filename")
    return safe


@app.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Upload one or more `.md`/`.txt`/`.pdf`/`.docx` files and ingest them.

    This does not implement its own ingestion logic — every accepted file
    is written into the configured upload directory (`data/uploads/`,
    never `data/corpus/`), then
    `app.ingestion.ingest_corpus()` (the exact same function
    `python -m app.ingestion` calls) does the loading, chunking, hashing,
    embedding, upserting, and stale-vector cleanup. See
    docs/architecture.md, "Upload architecture", for why this is one
    pipeline and not two.

    Per-file validation happens before any ingestion runs, so a batch with
    some bad files still ingests the good ones and reports exactly which
    files succeeded/failed and why (see docs/architecture.md, "Upload
    security"):
    - extension must be one of SUPPORTED_EXTENSIONS
    - filename is sanitized (`_safe_filename`) — no path traversal
    - empty files are rejected
    - files over MAX_UPLOAD_FILE_SIZE_BYTES are rejected without ever
      loading the whole file into memory (read is capped at limit + 1 byte)
    - a file that's saved but turns out corrupt/unreadable (bad PDF/DOCX)
      is caught immediately via `load_text` (the same extraction ingestion
      itself will run) and removed, rather than left as a broken file that
      would only fail later, mid-ingestion, for the whole batch

    Uploaded files are written to `data/uploads/` (the mutable runtime
    knowledge base), never to `data/corpus/` (the immutable, shipped
    assignment corpus) — see docs/architecture.md "Two knowledge bases, one
    logical corpus". `ingest_corpus()` still ingests both directories
    together on every call, so an uploaded document becomes retrievable by
    `/ask` immediately, with citations pointing at its real filename.
    """
    settings = get_settings()
    upload_dir = _resolve_upload_dir(settings)
    upload_dir.mkdir(parents=True, exist_ok=True)
    resolved_upload_dir = upload_dir.resolve()
    max_bytes = settings.max_upload_file_size_bytes

    accepted: list[str] = []
    rejected: list[UploadRejection] = []

    for upload in files:
        dest: Path | None = None
        try:
            safe_name = _safe_filename(upload.filename)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"unsupported file type '{suffix}' "
                    f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
                )

            data = await upload.read(max_bytes + 1)
            if not data:
                raise ValueError("file is empty")
            if len(data) > max_bytes:
                raise ValueError(
                    f"file exceeds the {max_bytes // (1024 * 1024)}MB upload limit"
                )

            dest = upload_dir / safe_name
            if not dest.resolve().is_relative_to(resolved_upload_dir):
                # Should be unreachable given _safe_filename never returns a
                # path with separators, but this is the last line of
                # defense against a path escaping the upload directory.
                raise ValueError("resolved destination is outside the upload directory")

            dest.write_bytes(data)

            # Catch a corrupt/unreadable file (bad PDF, bad DOCX, ...) here,
            # per-file, with the exact extraction ingest_corpus() will run
            # anyway — rather than letting it surface later as a SystemExit
            # that would fail the whole batch's ingestion.
            load_text(dest)

            accepted.append(safe_name)

        except (LoaderError, ValueError) as exc:
            if dest is not None and dest.exists():
                dest.unlink(missing_ok=True)
            rejected.append(
                UploadRejection(filename=upload.filename or "(unnamed)", error=str(exc))
            )
        except Exception as exc:  # noqa: BLE001
            # Never silently swallow — always report which file and why.
            _log.exception("unexpected error handling upload %s", upload.filename)
            if dest is not None and dest.exists():
                dest.unlink(missing_ok=True)
            rejected.append(
                UploadRejection(
                    filename=upload.filename or "(unnamed)",
                    error=f"unexpected error: {exc}",
                )
            )
        finally:
            await upload.close()

    if not accepted:
        return UploadResponse(status="error", files=[], rejected=rejected)

    try:
        summary = ingest_corpus(settings=settings)
    except SystemExit as exc:  # pragma: no cover - unreachable: accepted is non-empty
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except _UPSTREAM_ERRORS as exc:
        _log.warning(
            "upstream provider error on /upload: %s: %s", type(exc).__name__, exc
        )
        raise HTTPException(
            status_code=503,
            detail=f"Upstream service unavailable ({type(exc).__name__}). "
            f"Files were saved to data/uploads/ but not yet indexed; "
            f"re-run ingestion (or retry the upload) once the provider recovers.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return UploadResponse(
        status="success" if not rejected else "partial_success",
        files=accepted,
        rejected=rejected,
        chunks_created=summary["changed"],
        vectors_upserted=summary["upserted"],
        unchanged_chunks=summary["skipped_unchanged"],
        stale_vectors_deleted=summary["stale_deleted"],
        total_chunks=summary["chunks"],
        namespace_vector_count=summary["namespace_vector_count"],
    )


_RECURSION_REFUSAL = (
    "I cannot find the answer to this question in the provided documents."
)


def _short(exc: Exception, limit: int = 300) -> str:
    """Truncate an exception's string form.

    Avoids ever leaking a huge or sensitive payload.
    """
    text = str(exc)

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "…"
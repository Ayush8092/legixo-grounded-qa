"""Ingestion pipeline: corpus + upload files -> chunks -> embeddings -> Pinecone.

Usage:
    python -m app.ingestion            # idempotent: same chunk IDs are upserted in place
    python -m app.ingestion --reset    # wipe the namespace first, then ingest fresh

Supported input formats: `.md`, `.txt`, `.pdf`, `.docx` (see app/loaders.py).
`README.txt` (and any `README.md`) is corpus metadata, not a knowledge
document, and is never embedded (see docs/architecture.md).

Two knowledge bases, one logical corpus (see docs/architecture.md, "Two
knowledge bases, one logical corpus", and Prompt_11.docx): `data/corpus/`
is the immutable, shipped assignment corpus; `data/uploads/` is the
mutable, runtime knowledge base `POST /upload` writes into (see
`app/main.py`). Every ingestion run chunks and reconciles **both
directories together** via `app.chunking.chunk_directories` — never as two
separate `ingest_corpus()` calls. This matters: reconciling the two
directories independently would make each run see the *other* directory's
vectors as absent from "its" current chunk set and delete them as stale.
Treating `corpus_dir + upload_dir` as the one logical corpus for every
run's stale-vector computation is what prevents that.

Running ingestion twice does not create duplicate vectors: chunk IDs are
deterministic (`<source_root>::<filename-with-extension>::<section-slug>::<index>`
— see `app/chunking.py`'s `chunk_file` for why the source root and the full
filename, not just the stem, are both baked into the ID), so Pinecone
upserts land on the same vectors instead of creating duplicates. On top of
that, unchanged chunks are skipped entirely before the embedding call (see
`vectorstore.fetch_existing_hashes`) — re-running ingestion after editing
one document only re-embeds that document's chunks, not the whole corpus.

Stale-vector reconciliation: deleting a file (from either directory), or
removing a section from a file that survives, both leave old vectors behind
in Pinecone unless we explicitly clean up. Every ingestion run (outside of
`--reset`, which already wipes everything) computes:

    stale_ids = <every ID currently in the namespace> - <current logical corpus's chunk IDs>

where "current logical corpus" means corpus_dir + upload_dir chunked
together in this same run — and deletes exactly those IDs — never
`delete_all`, never anything outside the configured namespace. See
`vectorstore.list_all_ids` / `vectorstore.delete_vectors` for the mechanics
and the ownership reasoning.

`ingest_corpus()` is the one reusable ingestion entry point — both this
module's CLI (`main()` -> `run()`) and `app/main.py`'s `POST /upload`
endpoint call it, so there is exactly one
loader -> chunk (both dirs) -> hash -> embed -> Pinecone -> stale-cleanup
pipeline, not two.
"""

import argparse
from pathlib import Path

from app import vectorstore
from app.chunking import chunk_directories
from app.clients import ensure_index, get_embeddings, get_pinecone
from app.config import Settings, get_settings
from app.loaders import SUPPORTED_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_corpus_dir(settings: Settings) -> Path:
    path = Path(settings.corpus_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_upload_dir(settings: Settings) -> Path:
    path = Path(settings.upload_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ingest_corpus(
    reset: bool = False,
    settings: Settings | None = None,
    corpus_dir: Path | None = None,
    upload_dir: Path | None = None,
) -> dict:
    """Run the full ingestion pipeline once, across BOTH `corpus_dir` and
    `upload_dir` together. Returns a small summary dict.

    This is the single reusable ingestion service: the CLI (`run`, below)
    and the `POST /upload` API endpoint both call this and nothing else —
    neither implements its own loader/chunk/embed/upsert logic, and neither
    calls this once per directory (see module docstring for why that would
    be unsafe for stale-vector reconciliation).

    A missing/empty/nonexistent `upload_dir` (e.g. before any document has
    ever been uploaded) contributes zero chunks and does not raise — see
    `chunk_directories`/`chunk_corpus`'s graceful degradation — so this
    behaves identically to the pre-uploads pipeline when there are no
    uploads yet.
    """
    settings = settings or get_settings()
    corpus_dir = corpus_dir or _resolve_corpus_dir(settings)
    upload_dir = upload_dir or _resolve_upload_dir(settings)

    chunks = chunk_directories([(corpus_dir, "corpus"), (upload_dir, "uploads")])
    if not chunks:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise SystemExit(
            f"No ingestable files ({supported}) found in {corpus_dir} or {upload_dir}"
        )

    current_chunk_ids = {c.chunk_id for c in chunks}

    pc = get_pinecone(settings)
    index = ensure_index(pc, settings)

    if reset:
        vectorstore.delete_namespace(pc, settings)
        existing_hashes: dict[str, str] = {}
    else:
        existing_hashes = vectorstore.fetch_existing_hashes(
            index, settings, [c.chunk_id for c in chunks]
        )

    changed = [c for c in chunks if existing_hashes.get(c.chunk_id) != c.content_hash]
    skipped = len(chunks) - len(changed)

    upserted = 0
    if changed:
        embeddings = get_embeddings(settings)
        vectors = embeddings.embed_documents([c.text for c in changed])
        upserted = vectorstore.upsert_chunks(index, settings, changed, vectors)

    # --- Stale-vector reconciliation -----------------------------------
    # `current_chunk_ids` already spans BOTH directories (computed above
    # from the combined `chunks` list), so this one reconciliation pass
    # naturally can't mistake one directory's vectors for the other's
    # stale leftovers — there is no second, independent reconciliation
    # call to get wrong.
    if reset:
        # --reset already wiped the whole namespace above, so nothing
        # stale can remain; skip the extra list/delete round-trip.
        stale_deleted = 0
        stale_chunk_ids: list[str] = []
    else:
        existing_ids = vectorstore.list_all_ids(index, settings)
        stale_ids = sorted(existing_ids - current_chunk_ids)
        stale_deleted = vectorstore.delete_vectors(index, settings, stale_ids)
        stale_chunk_ids = stale_ids

    total = vectorstore.vector_count(pc, settings)

    return {
        "corpus_dir": str(corpus_dir),
        "upload_dir": str(upload_dir),
        "chunks": len(chunks),
        "changed": len(changed),
        "skipped_unchanged": skipped,
        "upserted": upserted,
        "stale_deleted": stale_deleted,
        "stale_chunk_ids": stale_chunk_ids,
        "namespace_vector_count": total,
        "chunk_ids": [c.chunk_id for c in chunks],
        "changed_chunk_ids": [c.chunk_id for c in changed],
        "source_files": sorted({c.source_file for c in chunks}),
    }


def run(reset: bool = False, settings: Settings | None = None) -> dict:
    """Backward-compatible alias for `ingest_corpus` — this is the CLI's entry point."""
    return ingest_corpus(reset=reset, settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus + uploads into Pinecone.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete everything in the namespace before ingesting (also disables the unchanged-chunk skip).",
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = _resolve_corpus_dir(settings)
    upload_dir = _resolve_upload_dir(settings)
    print(f"Corpus:  {corpus_dir}  (immutable)")
    print(f"Uploads: {upload_dir}  (mutable)")
    print(f"Index:   {settings.pinecone_index_name}  Namespace: {settings.pinecone_namespace}")

    summary = run(reset=args.reset, settings=settings)

    print(f"Chunked corpus + uploads -> {summary['chunks']} chunks, {len(summary['source_files'])} files")
    if args.reset:
        print("Namespace cleared (--reset)")
    if summary["skipped_unchanged"]:
        print(f"Skipped {summary['skipped_unchanged']} unchanged chunks (content hash matched Pinecone)")
    print(f"Embedded + upserted {summary['upserted']} vectors")
    if summary["stale_deleted"]:
        print(f"Deleted {summary['stale_deleted']} stale vectors (no longer produced by corpus + uploads):")
        for chunk_id in summary["stale_chunk_ids"]:
            print(f"  - {chunk_id}")
    print(f"Namespace now holds {summary['namespace_vector_count']} vectors")
    for chunk_id in summary["chunk_ids"]:
        marker = "•" if chunk_id in summary["changed_chunk_ids"] else "="
        print(f"  {marker} {chunk_id}")
    print("\n(• = embedded this run   = = unchanged, skipped)")


if __name__ == "__main__":
    main()

"""Pinecone read/write helpers: upsert, query, reset, stats.

`clients.py` owns *connecting*; this module owns *what we do* with the
index once connected. Kept separate so ingestion (writes) and retrieval
(reads) both go through the same, tested upsert/query shape.
"""

from app.chunking import Chunk

UPSERT_BATCH_SIZE = 100


def delete_namespace(pc, settings) -> None:
    """Wipe the configured namespace. Used by `python -m app.ingestion --reset`."""
    index = pc.Index(settings.pinecone_index_name)
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
    except Exception:
        # A brand-new index has no namespace yet — nothing to delete.
        pass


def upsert_chunks(
    index, settings, chunks: list[Chunk], vectors: list[list[float]]
) -> int:
    """Upsert chunks with their embeddings.

    Chunk IDs are deterministic (see chunking.py), so re-running ingestion
    overwrites the same Pinecone records instead of creating duplicates —
    this is what makes ingestion idempotent.
    """
    records = [
        {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "document_title": chunk.document_title,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "text": chunk.text,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(
            vectors=records[start : start + UPSERT_BATCH_SIZE],
            namespace=settings.pinecone_namespace,
        )
    return len(records)


def query(index, settings, vector: list[float], top_k: int) -> list[dict]:
    """Query the index; returns matches as plain dicts with score + metadata."""
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )
    matches = result.get("matches") if isinstance(result, dict) else result.matches
    out = []
    for m in matches:
        meta = m["metadata"] if isinstance(m, dict) else m.metadata
        mid = m["id"] if isinstance(m, dict) else m.id
        score = m["score"] if isinstance(m, dict) else m.score
        out.append(
            {
                "chunk_id": meta.get("chunk_id", mid),
                "score": float(score),
                "source_file": meta.get("source_file", ""),
                "document_title": meta.get("document_title", ""),
                "section": meta.get("section", ""),
                "text": meta.get("text", ""),
            }
        )
    return out


def fetch_existing_hashes(index, settings, chunk_ids: list[str]) -> dict[str, str]:
    """Look up `content_hash` metadata for chunk_ids already stored in Pinecone.

    Used by ingestion to skip re-embedding chunks whose content hasn't
    changed since the last run (Improvement 5) — the corpus is small enough
    that this isn't required for correctness, but it does cut Gemini
    embedding calls (and free-tier quota) on repeat ingestion runs where
    only one or two documents changed. `index.fetch` is read-only and safe
    to call even on an empty/brand-new namespace.

    Returns {chunk_id: content_hash} for whichever of the requested IDs
    already exist; missing IDs are simply absent from the result (treated
    as "new" by the caller).
    """
    if not chunk_ids:
        return {}
    hashes: dict[str, str] = {}
    for start in range(0, len(chunk_ids), UPSERT_BATCH_SIZE):
        batch = chunk_ids[start : start + UPSERT_BATCH_SIZE]
        try:
            result = index.fetch(ids=batch, namespace=settings.pinecone_namespace)
        except Exception:
            # Brand-new index/namespace, or a transient fetch error — treat
            # every chunk in this batch as new rather than failing ingestion.
            continue
        vectors = result.get("vectors") if isinstance(result, dict) else result.vectors
        for chunk_id, record in (vectors or {}).items():
            meta = record["metadata"] if isinstance(record, dict) else record.metadata
            content_hash = (meta or {}).get("content_hash")
            if content_hash:
                hashes[chunk_id] = content_hash
    return hashes


def vector_count(pc, settings) -> int:
    """Number of vectors currently stored in the configured namespace."""
    stats = pc.Index(settings.pinecone_index_name).describe_index_stats()
    namespaces = stats.get("namespaces") if isinstance(stats, dict) else stats.namespaces
    ns = (namespaces or {}).get(settings.pinecone_namespace)
    if ns is None:
        return 0
    return int(ns["vector_count"] if isinstance(ns, dict) else ns.vector_count)


def list_all_ids(index, settings) -> set[str]:
    """All vector IDs currently stored in the configured namespace.

    Chunk IDs and Pinecone vector IDs are the same string (see
    `upsert_chunks`: `"id": chunk.chunk_id`), so the result can be compared
    directly against the corpus's current chunk IDs to find stale vectors
    left behind by a deleted file or a restructured section — see
    `app/ingestion.py`'s reconciliation step.

    Scoped to `settings.pinecone_namespace` only, via the same
    `index.list(namespace=...)` call `delete_namespace` already uses for
    `--reset` — this project treats its configured namespace as exclusively
    its own (see `delete_namespace`), so every ID this returns is safe to
    reconcile against the current corpus. Nothing here ever touches another
    namespace or another index.

    Pinecone client versions/plans differ in whether `index.list(...)` is
    available (it requires a serverless index). If the installed client
    doesn't support it, this degrades to "nothing known" and prints a
    warning rather than raising — stale-vector cleanup is then skipped for
    this run, but insert/upsert/skip ingestion keeps working exactly as
    before this feature existed.
    """
    if not hasattr(index, "list"):
        print(
            "[ingestion] WARNING: this Pinecone client/index does not "
            "support index.list(); skipping stale-vector cleanup this run."
        )
        return set()

    ids: set[str] = set()
    try:
        for batch in index.list(namespace=settings.pinecone_namespace):
            ids.update(batch)
    except Exception as exc:
        print(
            f"[ingestion] WARNING: could not list existing vector IDs "
            f"({type(exc).__name__}: {exc}); skipping stale-vector cleanup this run."
        )
        return set()
    return ids


def delete_vectors(index, settings, chunk_ids: list[str]) -> int:
    """Delete specific vector IDs from the configured namespace, in batches.

    Always scoped to `settings.pinecone_namespace` and never `delete_all` —
    this can only ever remove vector IDs the caller explicitly names, never
    an entire namespace or index. `app/ingestion.py` is the only caller,
    and only ever passes IDs computed as `existing - current_chunk_ids`
    (see `list_all_ids`), so this never deletes a vector still produced by
    the current corpus.
    """
    if not chunk_ids:
        return 0
    for start in range(0, len(chunk_ids), UPSERT_BATCH_SIZE):
        batch = chunk_ids[start : start + UPSERT_BATCH_SIZE]
        index.delete(ids=batch, namespace=settings.pinecone_namespace)
    return len(chunk_ids)

"""Markdown-aware chunking with deterministic, context-rich chunks.

Design decisions (see docs/architecture.md for the full rationale):

- Corpus files are small legal-style notes with a clear structure: an H1
  title, a short block of "front matter" (bold key/value lines such as
  **Employee:** ...), and one or more H2 sections. We parse that structure
  instead of splitting on a fixed character window, so a chunk never starts
  or ends mid-clause.
- Every chunk is prefixed with the document title and front matter
  (Improvement 2 — "context-rich chunks"). A chunk that only contains
  "Either party may end this agreement by giving 60 days written notice."
  is ambiguous once it is pulled out of the document; a chunk that also
  carries "Employment agreement excerpt — Bluecrest Analytics / Employee:
  Priya Nambiar / Employer: Bluecrest Analytics LLP" is self-contained and
  gives the LLM and the citation snippet enough context to be useful on
  their own.
- Chunk IDs are deterministic: `<file-stem>::<section-slug>::<index>`.
  Re-running ingestion recomputes the same IDs, so Pinecone upserts land on
  the same vectors instead of creating duplicates (idempotent ingestion).
- Oversized sections are split into overlapping windows only as a fallback
  cap — the corpus is small and most sections fit in a single chunk, so we
  avoid unnecessary fragmentation.

This module is pure (no network calls) so it is fully unit-testable offline.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from app.loaders import EXCLUDED_FILENAMES, SUPPORTED_EXTENSIONS, load_text

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_root: str
    source_file: str
    document_title: str
    section: str
    chunk_index: int
    text: str
    content_hash: str


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _content_hash(text: str) -> str:
    """Short, stable hash of chunk text — used to detect unchanged chunks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_document(content: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Split a corpus file into (title, front_matter, [(section_title, body), ...]).

    The title is the first `# ` heading. Everything between the title and the
    first `## ` heading is treated as front matter (e.g. bold Employee/Employer
    lines) and is prepended to every section chunk for context. Sections are
    delimited by `## ` headings.
    """
    lines = content.splitlines()
    title = ""
    front_matter_lines: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, current_lines))
            elif "".join(front_matter_lines).strip():
                pass  # front matter collected separately, nothing to flush
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is None:
            front_matter_lines.append(line)
        else:
            current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, current_lines))

    front_matter = "\n".join(front_matter_lines).strip()
    body_sections = [(t, "\n".join(b).strip()) for t, b in sections]
    return title, front_matter, body_sections


def _split_oversized(text: str) -> list[str]:
    """Fallback size cap: split long sections into overlapping windows."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + MAX_CHUNK_CHARS])
        start += MAX_CHUNK_CHARS - OVERLAP_CHARS
    return parts


def _context_prefix(title: str, front_matter: str, section: str) -> str:
    lines = []
    if title:
        lines.append(f"Document: {title}")
    if front_matter:
        lines.append(front_matter)
    if section:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


def chunk_file(path: Path, source_root: str = "corpus") -> list[Chunk]:
    """Chunk a single corpus file (`.md`, `.txt`, `.pdf`, or `.docx`).

    Text extraction is format-specific (see `app/loaders.py`); everything
    after that — title/front-matter/section parsing, oversized-section
    splitting, chunk ID generation — is identical regardless of source
    format, which is what keeps this one pipeline instead of four.

    `source_root` identifies which logical knowledge base this file came
    from (`"corpus"` for the immutable `data/corpus/`, `"uploads"` for the
    runtime `data/uploads/`) and is baked into the chunk_id — see the
    chunk_id shape note below.
    """
    content = load_text(path)
    stem = path.stem
    title, front_matter, sections = _parse_document(content)

    if not sections:
        # No `##` sections at all (shouldn't happen in this corpus, but
        # degrade gracefully rather than dropping the file).
        sections = [("header", content.strip())]

    chunks: list[Chunk] = []
    for section_title, body in sections:
        if not body:
            continue
        slug = _slugify(section_title)
        prefix = _context_prefix(title, front_matter, section_title)
        for n, part in enumerate(_split_oversized(body)):
            text = f"{prefix}\n\n{part}".strip() if prefix else part
            chunks.append(
                Chunk(
                    # Shape: <source_root>::<filename-with-extension>::<section-slug>::<index>
                    #
                    # Two things this format guards against (see
                    # docs/architecture.md "Chunk identity across corpus and
                    # uploads"):
                    #   1. The *full filename with extension* (not just the
                    #      stem) is used, so `notes.txt` and `notes.docx`
                    #      never collide even if their content is identical
                    #      — a bare stem would have produced the same ID
                    #      for both.
                    #   2. `source_root` distinguishes a file in
                    #      data/corpus/ from a same-named file in
                    #      data/uploads/, so the two knowledge bases can be
                    #      chunked and reconciled together without ever
                    #      colliding on ID.
                    chunk_id=f"{source_root}::{path.name}::{slug}::{n}",
                    source_root=source_root,
                    source_file=path.name,
                    document_title=title or stem,
                    section=section_title,
                    chunk_index=n,
                    text=text,
                    content_hash=_content_hash(text),
                )
            )
    return chunks


def chunk_corpus(corpus_dir: Path, source_root: str = "corpus") -> list[Chunk]:
    """Chunk every supported file (`.md`/`.txt`/`.pdf`/`.docx`) in a directory.

    `README.txt` (and any `README.md`) is corpus documentation, not a
    knowledge document, and is never ingested — see `EXCLUDED_FILENAMES`.

    Files are ingested in sorted-by-name order (across all formats
    together) for deterministic, reproducible output.

    `source_root` is tagged onto every chunk_id produced from this
    directory (default `"corpus"`, matching `data/corpus/`; pass
    `source_root="uploads"` when chunking `data/uploads/` — see
    `app.ingestion.ingest_corpus`, which calls this once per directory and
    merges the results into one logical corpus).
    """
    if not corpus_dir.is_dir():
        return []

    paths = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.name.lower() not in EXCLUDED_FILENAMES
    )

    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_file(path, source_root=source_root))
    return chunks


def chunk_directories(directories: list[tuple[Path, str]]) -> list[Chunk]:
    """Chunk several directories together into one logical corpus.

    Used by `app.ingestion.ingest_corpus` to combine the immutable
    `data/corpus/` and the mutable `data/uploads/` into a single chunk set
    before computing which Pinecone vectors are current vs. stale — see
    docs/architecture.md "Unified ingestion across corpus and uploads" for
    why this must happen in one pass rather than two separate
    `chunk_corpus()` calls reconciled independently.

    `directories` is a list of `(path, source_root)` pairs, e.g.
    `[(corpus_dir, "corpus"), (upload_dir, "uploads")]`. A missing
    directory contributes zero chunks (same graceful-degradation behavior
    as `chunk_corpus`) rather than raising, so a fresh checkout with no
    `data/uploads/` yet works exactly like before uploads existed.

    Raises `ValueError` if two directories somehow produce the same
    chunk_id — this should be unreachable given `source_root` is baked
    into every ID, but it's cheap insurance against a future caller
    passing the same `source_root` twice for two different directories.
    """
    chunks: list[Chunk] = []
    seen: dict[str, str] = {}  # chunk_id -> which directory produced it
    for directory, source_root in directories:
        for chunk in chunk_corpus(directory, source_root=source_root):
            if chunk.chunk_id in seen:
                raise ValueError(
                    f"chunk_id collision: '{chunk.chunk_id}' was produced by both "
                    f"{seen[chunk.chunk_id]} and {directory} — this should be "
                    f"impossible when each directory uses a distinct source_root."
                )
            seen[chunk.chunk_id] = str(directory)
            chunks.append(chunk)
    return chunks


def preview(chunks: list[Chunk]) -> str:
    lines = [
        f"{c.chunk_id}  ({len(c.text)} chars)  [{c.source_file} / {c.section}]"
        for c in chunks
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    corpus = Path(__file__).resolve().parent.parent / "data" / "corpus"
    all_chunks = chunk_corpus(corpus)
    print(preview(all_chunks))
    print(f"\n{len(all_chunks)} chunks from {corpus}")
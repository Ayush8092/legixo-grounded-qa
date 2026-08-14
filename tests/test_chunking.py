"""Offline unit tests for app.chunking. No network calls, no API keys required."""

from pathlib import Path

import pytest

from app.chunking import chunk_corpus, chunk_directories, chunk_file
from app.loaders import LoaderError

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_readme_txt_is_never_ingested():
    """README.txt is corpus documentation, not a knowledge document, and
    must never be ingested — but this does NOT mean .txt files in general
    are unsupported (see app/loaders.py; ordinary .txt documents ARE
    ingestible, and test_chunk_corpus_ingests_a_mix_of_md_and_txt_files
    below covers that on an isolated tmp_path corpus). This test only
    checks the one specific exclusion, against the real corpus directory,
    where README.txt is the only .txt file expected to exist at all."""
    chunks = chunk_corpus(CORPUS_DIR)
    assert all(c.source_file.lower() != "readme.txt" for c in chunks)


def test_only_the_six_known_source_files_appear():
    """This checks the real `data/corpus/` directory, not an isolated
    tmp_path — so it will (correctly) fail if that directory has been
    manually contaminated, e.g. by testing `POST /upload` locally against
    the real corpus dir instead of a throwaway one, or by dropping a
    scratch file in for a quick manual check. That is a real signal, not a
    flaky test: `data/corpus/` is expected to hold exactly the six
    assignment documents. If this fails, run `ls data/corpus/` and remove
    anything that isn't one of the six files below (or move ad-hoc manual
    testing to a separate directory / tmp_path) rather than editing this
    test's expectations.
    """
    expected = {
        "01_matter_memo_arvind_v_northfield.md",
        "02_employment_agreement_excerpt.md",
        "03_hearing_notice_template.md",
        "04_statute_style_excerpt_fictional.md",
        "05_counsel_notes_settlement.md",
        "06_property_lease_clause.md",
    }
    chunks = chunk_corpus(CORPUS_DIR)
    actual = {c.source_file for c in chunks}
    extra = actual - expected
    missing = expected - actual
    assert not extra, (
        f"data/corpus/ contains unexpected file(s) not part of the six-document "
        f"assignment corpus: {sorted(extra)} — likely leftover from manual "
        f"testing (e.g. POST /upload against the real corpus dir). Remove "
        f"them from data/corpus/ rather than changing this test."
    )
    assert not missing, f"data/corpus/ is missing expected file(s): {sorted(missing)}"


def test_chunk_ids_are_deterministic_across_runs():
    first = [c.chunk_id for c in chunk_corpus(CORPUS_DIR)]
    second = [c.chunk_id for c in chunk_corpus(CORPUS_DIR)]
    assert first == second
    assert len(first) == len(set(first)), "chunk IDs must be unique"


def test_chunk_id_shape_is_source_section_index():
    chunks = chunk_corpus(CORPUS_DIR)
    for c in chunks:
        parts = c.chunk_id.split("::")
        assert len(parts) == 4
        source_root, filename, _slug, index = parts
        assert source_root == "corpus"
        assert filename == c.source_file
        assert index.isdigit()


def test_chunk_id_includes_source_root_and_full_filename_with_extension():
    """Regression guard: chunk IDs must carry both the source directory
    identity (corpus vs. uploads) and the full filename INCLUDING
    extension, not just the stem — otherwise `notes.txt` and `notes.docx`,
    or a same-named file in data/corpus/ vs data/uploads/, would collide."""
    chunks = chunk_corpus(CORPUS_DIR, source_root="corpus")
    sample = next(c for c in chunks if c.source_file == "02_employment_agreement_excerpt.md")
    assert sample.chunk_id.startswith("corpus::02_employment_agreement_excerpt.md::")
    assert sample.source_root == "corpus"


def test_same_stem_different_extension_produces_distinct_chunk_ids(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "# Notes\n\n## Leave policy\n\nEmployees receive 24 paid leave days.\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.docx.md").write_text(  # stand-in text file with a similar stem
        "# Notes\n\n## Leave policy\n\nEmployees receive 24 paid leave days.\n",
        encoding="utf-8",
    )
    chunks = chunk_corpus(tmp_path)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "same-stem, different-extension files must not collide"


def test_same_filename_different_source_root_produces_distinct_chunk_ids(tmp_path):
    """The exact scenario from the docx: uploading a file with the same
    name as an official corpus document must not collide with it once both
    are chunked together (via distinct source_root tags)."""
    (tmp_path / "shared.md").write_text(
        "# Shared\n\n## Section\n\nSame filename, different knowledge base.\n",
        encoding="utf-8",
    )
    corpus_chunks = chunk_corpus(tmp_path, source_root="corpus")
    upload_chunks = chunk_corpus(tmp_path, source_root="uploads")
    corpus_ids = {c.chunk_id for c in corpus_chunks}
    upload_ids = {c.chunk_id for c in upload_chunks}
    assert corpus_ids.isdisjoint(upload_ids)


def test_employment_agreement_has_notice_period_section():
    path = CORPUS_DIR / "02_employment_agreement_excerpt.md"
    chunks = chunk_file(path)
    sections = {c.section for c in chunks}
    assert "Notice period" in sections
    notice_chunk = next(c for c in chunks if c.section == "Notice period")
    assert "60 days" in notice_chunk.text


def test_chunks_carry_document_title_and_front_matter_context():
    """Improvement 2: a chunk should be understandable on its own."""
    path = CORPUS_DIR / "02_employment_agreement_excerpt.md"
    chunks = chunk_file(path)
    notice_chunk = next(c for c in chunks if c.section == "Notice period")
    assert "Bluecrest Analytics" in notice_chunk.text
    assert "Priya Nambiar" in notice_chunk.text
    assert notice_chunk.document_title


def test_rerunning_ingestion_produces_the_same_content_hashes():
    first = {c.chunk_id: c.content_hash for c in chunk_corpus(CORPUS_DIR)}
    second = {c.chunk_id: c.content_hash for c in chunk_corpus(CORPUS_DIR)}
    assert first == second


def test_oversized_section_is_split_with_overlap(tmp_path):
    long_body = "word " * 1000  # well over MAX_CHUNK_CHARS
    doc = f"# Title\n\n## Big section\n\n{long_body}\n"
    path = tmp_path / "07_synthetic.md"
    path.write_text(doc, encoding="utf-8")

    chunks = chunk_file(path)
    assert len(chunks) > 1
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(ids)


def test_missing_corpus_dir_yields_no_chunks(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert chunk_corpus(empty_dir) == []


def test_nonexistent_corpus_dir_yields_no_chunks_instead_of_raising(tmp_path):
    """A corpus_dir path that was never created (vs. an existing empty dir,
    covered above) must degrade gracefully rather than raising — the old
    glob()-based implementation didn't raise on a missing directory, and
    the iterdir()-based multi-format implementation must preserve that."""
    never_created = tmp_path / "does-not-exist"
    assert chunk_corpus(never_created) == []


def test_readme_txt_stays_excluded_once_txt_ingestion_is_supported(tmp_path):
    """Regression guard for the multi-format change: now that .txt is a
    supported, ingestable format, README.txt must still be excluded by
    name, not just because '.txt' used to be unsupported."""
    (tmp_path / "README.txt").write_text("Corpus metadata, not a document.", encoding="utf-8")
    (tmp_path / "01_real_doc.txt").write_text("Actual document content.", encoding="utf-8")

    chunks = chunk_corpus(tmp_path)
    source_files = {c.source_file for c in chunks}
    assert "README.txt" not in source_files
    assert "01_real_doc.txt" in source_files


def test_chunk_corpus_ingests_a_mix_of_md_and_txt_files(tmp_path):
    (tmp_path / "01_markdown.md").write_text(
        "# Markdown Doc\n\n## Section One\n\nMarkdown body text.\n", encoding="utf-8"
    )
    (tmp_path / "02_plaintext.txt").write_text(
        "Plain text document with no markdown headings at all.", encoding="utf-8"
    )

    chunks = chunk_corpus(tmp_path)
    source_files = {c.source_file for c in chunks}
    assert source_files == {"01_markdown.md", "02_plaintext.txt"}

    txt_chunk = next(c for c in chunks if c.source_file == "02_plaintext.txt")
    assert "Plain text document" in txt_chunk.text
    # No '##' headings in a .txt file -> falls back to the same single
    # "header" section chunking.py already uses for a .md file with no
    # '##' sections.
    assert txt_chunk.section == "header"


def test_chunk_file_raises_loader_error_for_unsupported_extension(tmp_path):
    path = tmp_path / "notes.xlsx"
    path.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(LoaderError):
        chunk_file(path)


# ---------------------------------------------------------------------
# chunk_directories: merging corpus + uploads into one logical corpus
# ---------------------------------------------------------------------


def test_chunk_directories_merges_both_directories(tmp_path):
    corpus_dir = tmp_path / "corpus"
    upload_dir = tmp_path / "uploads"
    corpus_dir.mkdir()
    upload_dir.mkdir()
    (corpus_dir / "official.md").write_text(
        "# Official\n\n## Section\n\nOfficial content.\n", encoding="utf-8"
    )
    (upload_dir / "uploaded.md").write_text(
        "# Uploaded\n\n## Section\n\nUploaded content.\n", encoding="utf-8"
    )

    chunks = chunk_directories([(corpus_dir, "corpus"), (upload_dir, "uploads")])

    source_files = {c.source_file for c in chunks}
    roots = {c.source_root for c in chunks}
    assert source_files == {"official.md", "uploaded.md"}
    assert roots == {"corpus", "uploads"}


def test_chunk_directories_tolerates_a_missing_directory(tmp_path):
    """A fresh checkout with no data/uploads/ yet must not raise — it just
    contributes zero chunks, same as chunk_corpus on a missing directory."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "official.md").write_text(
        "# Official\n\n## Section\n\nContent.\n", encoding="utf-8"
    )
    never_created_upload_dir = tmp_path / "does-not-exist"

    chunks = chunk_directories(
        [(corpus_dir, "corpus"), (never_created_upload_dir, "uploads")]
    )

    assert len(chunks) == 1
    assert chunks[0].source_root == "corpus"


def test_chunk_directories_ids_never_collide_across_two_directories(tmp_path):
    """Same filename AND same section content in both directories — the
    resulting chunk_ids must still be fully distinct, and each chunk must
    retain its own source_root."""
    corpus_dir = tmp_path / "corpus"
    upload_dir = tmp_path / "uploads"
    corpus_dir.mkdir()
    upload_dir.mkdir()
    identical_content = "# Doc\n\n## Section\n\nIdentical text in both places.\n"
    (corpus_dir / "doc.md").write_text(identical_content, encoding="utf-8")
    (upload_dir / "doc.md").write_text(identical_content, encoding="utf-8")

    chunks = chunk_directories([(corpus_dir, "corpus"), (upload_dir, "uploads")])

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # no collision despite identical filename + content
    roots_by_id = {c.chunk_id: c.source_root for c in chunks}
    assert set(roots_by_id.values()) == {"corpus", "uploads"}
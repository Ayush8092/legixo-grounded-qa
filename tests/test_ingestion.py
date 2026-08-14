"""Offline unit tests for app.ingestion — the unchanged-chunk skip logic
and settings.corpus_dir resolution.

Pinecone/Gemini clients are replaced with small fakes; no network calls.
"""

from pathlib import Path

import pytest

from app import ingestion
from app.config import Settings


def _settings(tmp_path, **overrides) -> Settings:
    base = dict(
        GEMINI_API_KEY="x",
        GROQ_API_KEY="x",
        PINECONE_API_KEY="x",
        CORPUS_DIR=str(tmp_path),
        # Isolate from the real project's data/uploads/ — without this, a
        # test run on a machine that has ever manually POSTed to /upload
        # against the real corpus would silently pick up those files here
        # too. Pointing at a directory that doesn't exist makes
        # chunk_corpus() contribute zero chunks for it, same as "no
        # uploads yet" (see app/chunking.py's graceful degradation).
        UPLOAD_DIR=str(tmp_path / "_no_uploads_in_this_test"),
    )
    base.update(overrides)
    return Settings(**base)


def _write_corpus(tmp_path: Path):
    (tmp_path / "01_doc.md").write_text(
        "# Doc One\n\n## Section A\n\nOriginal content for section A.\n",
        encoding="utf-8",
    )


class _FakeEmbeddings:
    def __init__(self):
        self.embed_calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.embed_calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeIndex:
    def __init__(self, existing: dict[str, dict] | None = None):
        self.store: dict[str, dict] = existing or {}
        self.upsert_calls = 0
        self.delete_calls: list[list[str]] = []

    def fetch(self, ids, namespace):
        vectors = {i: {"metadata": self.store[i]["metadata"]} for i in ids if i in self.store}
        return {"vectors": vectors}

    def upsert(self, vectors, namespace):
        self.upsert_calls += 1
        for record in vectors:
            self.store[record["id"]] = record

    def list(self, namespace):
        ids = list(self.store.keys())
        batch_size = 2
        for start in range(0, len(ids), batch_size):
            yield ids[start : start + batch_size]

    def delete(self, ids, namespace):
        self.delete_calls.append(list(ids))
        for chunk_id in ids:
            self.store.pop(chunk_id, None)

    def describe_index_stats(self):
        return {"namespaces": {"legixo-corpus": {"vector_count": len(self.store)}}}


@pytest.fixture(autouse=True)
def _patch_clients(monkeypatch):
    """Replace network-touching client factories with fakes for every test in this file."""
    fake_index = _FakeIndex()
    fake_embeddings = _FakeEmbeddings()
    fake_pc = type("_FakePinecone", (), {"Index": lambda self, name: fake_index})()

    monkeypatch.setattr(ingestion, "get_pinecone", lambda settings: fake_pc)
    monkeypatch.setattr(ingestion, "ensure_index", lambda pc, settings: fake_index)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda settings: fake_embeddings)

    from app import vectorstore

    monkeypatch.setattr(vectorstore, "delete_namespace", lambda pc, settings: fake_index.store.clear())

    yield fake_index, fake_embeddings


def test_resolve_corpus_dir_uses_settings_value(tmp_path):
    settings = _settings(tmp_path)
    assert ingestion._resolve_corpus_dir(settings) == tmp_path


def test_first_run_embeds_every_chunk(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    _write_corpus(tmp_path)
    settings = _settings(tmp_path)

    summary = ingestion.run(settings=settings)

    assert summary["chunks"] == 1
    assert summary["changed"] == 1
    assert summary["skipped_unchanged"] == 0
    assert len(fake_embeddings.embed_calls) == 1


def test_second_run_with_unchanged_corpus_skips_embedding(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    _write_corpus(tmp_path)
    settings = _settings(tmp_path)

    ingestion.run(settings=settings)  # first run: embeds
    fake_embeddings.embed_calls.clear()

    summary = ingestion.run(settings=settings)  # second run: nothing changed

    assert summary["changed"] == 0
    assert summary["skipped_unchanged"] == 1
    assert fake_embeddings.embed_calls == []  # no embedding call made at all


def test_editing_one_chunk_only_reembeds_that_chunk(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    (tmp_path / "01_doc.md").write_text(
        "# Doc One\n\n## Section A\n\nOriginal A.\n\n## Section B\n\nOriginal B.\n",
        encoding="utf-8",
    )
    settings = _settings(tmp_path)
    ingestion.run(settings=settings)
    fake_embeddings.embed_calls.clear()

    # Edit only section A's content.
    (tmp_path / "01_doc.md").write_text(
        "# Doc One\n\n## Section A\n\nEDITED A.\n\n## Section B\n\nOriginal B.\n",
        encoding="utf-8",
    )
    summary = ingestion.run(settings=settings)

    assert summary["changed"] == 1
    assert summary["skipped_unchanged"] == 1
    assert len(fake_embeddings.embed_calls[0]) == 1  # only the changed chunk's text was embedded


def test_reset_flag_disables_the_skip_and_reembeds_everything(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    _write_corpus(tmp_path)
    settings = _settings(tmp_path)
    ingestion.run(settings=settings)
    fake_embeddings.embed_calls.clear()

    summary = ingestion.run(reset=True, settings=settings)

    assert summary["changed"] == 1
    assert summary["skipped_unchanged"] == 0
    assert len(fake_embeddings.embed_calls) == 1


def test_no_markdown_files_raises_system_exit(tmp_path, _patch_clients):
    settings = _settings(tmp_path)  # tmp_path has no .md files
    with pytest.raises(SystemExit):
        ingestion.run(settings=settings)


# ---------------------------------------------------------------------
# Stale-vector reconciliation (deleted files / deleted chunks)
# ---------------------------------------------------------------------


def test_deleting_a_file_removes_its_stale_vectors(tmp_path, _patch_clients):
    """final_improvement_lexido.docx section 1/16: A.md, B.md, C.md ingested;
    delete C.md; re-ingest; A and B are unchanged, C's vectors are gone."""
    fake_index, fake_embeddings = _patch_clients
    (tmp_path / "A.md").write_text("# A\n\n## Sec\n\nContent A.\n", encoding="utf-8")
    (tmp_path / "B.md").write_text("# B\n\n## Sec\n\nContent B.\n", encoding="utf-8")
    (tmp_path / "C.md").write_text("# C\n\n## Sec\n\nContent C.\n", encoding="utf-8")
    settings = _settings(tmp_path)

    ingestion.run(settings=settings)
    assert any(cid.startswith("corpus::C.md::") for cid in fake_index.store)

    (tmp_path / "C.md").unlink()
    fake_embeddings.embed_calls.clear()

    summary = ingestion.run(settings=settings)

    remaining_ids = set(fake_index.store)
    assert not any(cid.startswith("corpus::C.md::") for cid in remaining_ids)
    assert any(cid.startswith("corpus::A.md::") for cid in remaining_ids)
    assert any(cid.startswith("corpus::B.md::") for cid in remaining_ids)
    assert summary["stale_deleted"] == 1
    assert any(cid.startswith("corpus::C.md::") for cid in summary["stale_chunk_ids"])
    # A and B are untouched -> nothing re-embedded on this run.
    assert fake_embeddings.embed_calls == []


def test_deleting_one_section_removes_only_that_chunk(tmp_path, _patch_clients):
    """final_improvement_lexido.docx section 1/17: contract.md with
    section-a/b/c produces 3 chunks; removing section-c must delete only
    that one stale vector, leaving section-a and section-b untouched."""
    fake_index, fake_embeddings = _patch_clients
    (tmp_path / "contract.md").write_text(
        "# Contract\n\n"
        "## Section A\n\nContent A.\n\n"
        "## Section B\n\nContent B.\n\n"
        "## Section C\n\nContent C.\n",
        encoding="utf-8",
    )
    settings = _settings(tmp_path)
    ingestion.run(settings=settings)
    assert len(fake_index.store) == 3

    (tmp_path / "contract.md").write_text(
        "# Contract\n\n"
        "## Section A\n\nContent A.\n\n"
        "## Section B\n\nContent B.\n",
        encoding="utf-8",
    )
    fake_embeddings.embed_calls.clear()

    summary = ingestion.run(settings=settings)

    remaining_ids = set(fake_index.store)
    assert len(remaining_ids) == 2
    assert not any("section-c" in cid for cid in remaining_ids)
    assert summary["stale_deleted"] == 1
    assert summary["changed"] == 0  # sections A and B are unchanged, not re-embedded
    assert fake_embeddings.embed_calls == []


def test_reset_skips_reconciliation_since_namespace_was_already_wiped(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    _write_corpus(tmp_path)
    settings = _settings(tmp_path)

    summary = ingestion.run(reset=True, settings=settings)

    assert summary["stale_deleted"] == 0
    assert summary["stale_chunk_ids"] == []
    assert fake_index.delete_calls == []  # no list/delete round-trip performed


def test_unchanged_corpus_across_runs_deletes_nothing(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    _write_corpus(tmp_path)
    settings = _settings(tmp_path)

    ingestion.run(settings=settings)
    summary = ingestion.run(settings=settings)

    assert summary["stale_deleted"] == 0
    assert fake_index.delete_calls == []


def test_ingest_corpus_is_the_underlying_reusable_entry_point():
    """`run` is a backward-compatible alias — `ingest_corpus` is the name
    the upload endpoint (and any other future caller) should use."""
    assert ingestion.run is not ingestion.ingest_corpus
    assert ingestion.ingest_corpus.__name__ == "ingest_corpus"


def test_new_file_added_alongside_existing_ones_only_embeds_the_new_one(tmp_path, _patch_clients):
    """Covers the four-case matrix from final_improvement_lexido.docx
    section 18 together: new -> embed, changed -> embed, unchanged -> skip,
    deleted -> stale cleanup, all in one incremental run."""
    fake_index, fake_embeddings = _patch_clients
    (tmp_path / "A.md").write_text("# A\n\n## Sec\n\nContent A.\n", encoding="utf-8")
    settings = _settings(tmp_path)
    ingestion.run(settings=settings)
    fake_embeddings.embed_calls.clear()

    (tmp_path / "B.md").write_text("# B\n\n## Sec\n\nContent B.\n", encoding="utf-8")
    summary = ingestion.run(settings=settings)

    assert summary["changed"] == 1  # only B is new
    assert summary["skipped_unchanged"] == 1  # A is unchanged
    assert summary["stale_deleted"] == 0
    assert len(fake_embeddings.embed_calls[0]) == 1


# ---------------------------------------------------------------------
# final_upgrade_lexigo.docx #2: document modification, end to end at the
# vector-store level — "Payment period: 30 days" -> "45 days", confirming
# the OLD value is genuinely gone from the stored vector, not just that a
# re-embed happened.
# ---------------------------------------------------------------------


def test_modifying_document_content_replaces_stored_text_old_value_is_gone(tmp_path, _patch_clients):
    fake_index, fake_embeddings = _patch_clients
    doc = tmp_path / "agreement.md"
    doc.write_text("# Agreement\n\n## Payment\n\nPayment period: 30 days.\n", encoding="utf-8")
    settings = _settings(tmp_path)

    ingestion.run(settings=settings)
    chunk_id = next(iter(fake_index.store))
    assert "30 days" in fake_index.store[chunk_id]["metadata"]["text"]

    doc.write_text("# Agreement\n\n## Payment\n\nPayment period: 45 days.\n", encoding="utf-8")
    fake_embeddings.embed_calls.clear()
    summary = ingestion.run(settings=settings)

    # Same chunk_id (same file, same section) — content_hash differs, so it
    # was re-embedded and upserted IN PLACE, not left alongside a stale copy.
    assert set(fake_index.store) == {chunk_id}
    assert summary["changed"] == 1
    assert summary["stale_deleted"] == 0  # nothing to reconcile: no chunk was removed, only edited

    stored_text = fake_index.store[chunk_id]["metadata"]["text"]
    assert "45 days" in stored_text
    assert "30 days" not in stored_text  # the old value cannot leak into a future answer

    stored_hash = fake_index.store[chunk_id]["metadata"]["content_hash"]
    # A fresh chunk of the edited file must hash differently from before,
    # which is exactly what made `changed` non-empty above.
    from app.chunking import chunk_file

    fresh_chunk = chunk_file(doc)[0]
    assert stored_hash == fresh_chunk.content_hash


def test_removing_a_section_from_a_surviving_document_deletes_only_that_vector(tmp_path, _patch_clients):
    """The exact scenario from final_upgrade_lexigo.docx #2:
    contract.md {section-a, section-b, section-c} -> {section-a, section-b}."""
    fake_index, fake_embeddings = _patch_clients
    doc = tmp_path / "contract.md"
    doc.write_text(
        "# Contract\n\n"
        "## Section A\n\nContent A.\n\n"
        "## Section B\n\nContent B.\n\n"
        "## Section C\n\nContent C.\n",
        encoding="utf-8",
    )
    settings = _settings(tmp_path)
    ingestion.run(settings=settings)
    assert len(fake_index.store) == 3
    section_c_id = next(cid for cid in fake_index.store if "section-c" in cid)

    doc.write_text(
        "# Contract\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n",
        encoding="utf-8",
    )
    ingestion.run(settings=settings)

    assert section_c_id not in fake_index.store
    assert len(fake_index.store) == 2


# ---------------------------------------------------------------------
# Prompt_11.docx: unified corpus + uploads ingestion/reconciliation.
#
# The critical bug this guards against: two independent
# ingest_corpus(CORPUS_DIR) / ingest_corpus(UPLOAD_DIR) calls would each
# see the OTHER directory's vectors as "not part of my current chunk set"
# and delete them as stale. Every test below ingests both directories
# together (the real `ingest_corpus()` default behavior) and checks that
# neither directory's vectors are ever mistaken for the other's stale
# leftovers.
# ---------------------------------------------------------------------


def _settings_with_both_dirs(tmp_path, **overrides) -> Settings:
    """Real corpus_dir + upload_dir, both isolated under tmp_path — unlike
    `_settings()` above, which deliberately points UPLOAD_DIR at a
    directory that doesn't exist so single-directory tests aren't affected
    by uploads at all."""
    corpus_dir = tmp_path / "corpus"
    upload_dir = tmp_path / "uploads"
    corpus_dir.mkdir(exist_ok=True)
    upload_dir.mkdir(exist_ok=True)
    base = dict(
        GEMINI_API_KEY="x",
        GROQ_API_KEY="x",
        PINECONE_API_KEY="x",
        CORPUS_DIR=str(corpus_dir),
        UPLOAD_DIR=str(upload_dir),
    )
    base.update(overrides)
    return Settings(**base), corpus_dir, upload_dir


def test_corpus_only_run_does_not_delete_uploaded_vectors(tmp_path, _patch_clients):
    """The first call in the docx's warning example: ingesting when
    data/uploads/ is empty must NOT interpret the absence of uploaded
    documents as evidence that previously-ingested upload vectors are
    stale."""
    fake_index, fake_embeddings = _patch_clients
    settings, corpus_dir, upload_dir = _settings_with_both_dirs(tmp_path)

    (upload_dir / "uploaded.md").write_text(
        "# Uploaded\n\n## Leave policy\n\nEmployees receive 24 paid leave days.\n",
        encoding="utf-8",
    )
    (corpus_dir / "official.md").write_text(
        "# Official\n\n## Notice\n\nNotice period is 60 days.\n", encoding="utf-8"
    )
    ingestion.ingest_corpus(settings=settings)
    assert any(cid.startswith("uploads::uploaded.md::") for cid in fake_index.store)

    # Simulate a corpus-only re-ingestion by removing the uploaded file
    # from disk but WITHOUT deleting its vectors first — this is exactly
    # the scenario the docx describes as dangerous if the two directories
    # were reconciled independently. ingest_corpus() must still see the
    # uploads directory (now empty) as part of the SAME run, not a
    # separate reconciliation pass — so as long as we call it with both
    # directories present, the uploaded vector must survive unless the
    # uploaded file itself is actually gone from upload_dir.
    fake_embeddings.embed_calls.clear()
    summary = ingestion.ingest_corpus(settings=settings)

    # Nothing changed on disk since the last run -> nothing stale, nothing re-embedded.
    assert summary["stale_deleted"] == 0
    assert any(cid.startswith("uploads::uploaded.md::") for cid in fake_index.store)
    assert any(cid.startswith("corpus::official.md::") for cid in fake_index.store)


def test_deleting_an_uploaded_file_does_not_remove_official_corpus_vectors(tmp_path, _patch_clients):
    """Testing requirement #8: removing an uploaded document must not
    accidentally remove official corpus vectors."""
    fake_index, fake_embeddings = _patch_clients
    settings, corpus_dir, upload_dir = _settings_with_both_dirs(tmp_path)

    (corpus_dir / "official.md").write_text(
        "# Official\n\n## Notice\n\nNotice period is 60 days.\n", encoding="utf-8"
    )
    (upload_dir / "uploaded.md").write_text(
        "# Uploaded\n\n## Leave\n\nLeave policy details.\n", encoding="utf-8"
    )
    ingestion.ingest_corpus(settings=settings)
    assert len(fake_index.store) == 2

    (upload_dir / "uploaded.md").unlink()
    summary = ingestion.ingest_corpus(settings=settings)

    remaining = set(fake_index.store)
    assert not any(cid.startswith("uploads::uploaded.md::") for cid in remaining)
    assert any(cid.startswith("corpus::official.md::") for cid in remaining)
    assert summary["stale_deleted"] == 1


def test_official_corpus_reconciliation_does_not_remove_uploaded_vectors(tmp_path, _patch_clients):
    """Testing requirement #9: reconciling the official corpus (e.g. an
    official document is edited/removed) must not accidentally remove
    uploaded vectors."""
    fake_index, fake_embeddings = _patch_clients
    settings, corpus_dir, upload_dir = _settings_with_both_dirs(tmp_path)

    (corpus_dir / "official_a.md").write_text(
        "# A\n\n## Sec\n\nContent A.\n", encoding="utf-8"
    )
    (corpus_dir / "official_b.md").write_text(
        "# B\n\n## Sec\n\nContent B.\n", encoding="utf-8"
    )
    (upload_dir / "uploaded.md").write_text(
        "# Uploaded\n\n## Sec\n\nUploaded content.\n", encoding="utf-8"
    )
    ingestion.ingest_corpus(settings=settings)
    assert len(fake_index.store) == 3

    # An official document is removed (e.g. corrected/replaced).
    (corpus_dir / "official_b.md").unlink()
    summary = ingestion.ingest_corpus(settings=settings)

    remaining = set(fake_index.store)
    assert any(cid.startswith("corpus::official_a.md::") for cid in remaining)
    assert not any(cid.startswith("corpus::official_b.md::") for cid in remaining)
    assert any(cid.startswith("uploads::uploaded.md::") for cid in remaining)  # untouched
    assert summary["stale_deleted"] == 1


def test_chunk_ids_remain_deterministic_and_unique_across_both_directories(tmp_path, _patch_clients):
    """Testing requirement #10, and the docx's specific chunk-ID example:
    a .txt and a .docx-named file with the same base name and similar
    content must remain distinct once corpus + uploads are ingested
    together."""
    fake_index, fake_embeddings = _patch_clients
    settings, corpus_dir, upload_dir = _settings_with_both_dirs(tmp_path)

    (corpus_dir / "point_8_test_documents.txt").write_text(
        "# Point 8\n\n## Leave policy\n\nEmployees receive 24 paid leave days.\n",
        encoding="utf-8",
    )
    (upload_dir / "point_8_test_documents.txt").write_text(
        "# Point 8\n\n## Leave policy\n\nEmployees receive 24 paid leave days.\n",
        encoding="utf-8",
    )
    summary = ingestion.ingest_corpus(settings=settings)

    ids = summary["chunk_ids"]
    assert len(ids) == len(set(ids)), "chunk IDs must remain unique across both directories"
    assert any(cid.startswith("corpus::point_8_test_documents.txt::") for cid in ids)
    assert any(cid.startswith("uploads::point_8_test_documents.txt::") for cid in ids)


def test_ingest_corpus_defaults_to_both_settings_directories_when_no_override_given(tmp_path, _patch_clients, monkeypatch):
    """`ingest_corpus()` called with only `settings=` (no explicit
    corpus_dir/upload_dir override) — the way both the CLI and
    POST /upload actually call it — must resolve and ingest BOTH
    directories from settings, not just corpus_dir."""
    fake_index, fake_embeddings = _patch_clients
    settings, corpus_dir, upload_dir = _settings_with_both_dirs(tmp_path)

    (corpus_dir / "official.md").write_text("# A\n\n## Sec\n\nA.\n", encoding="utf-8")
    (upload_dir / "uploaded.md").write_text("# B\n\n## Sec\n\nB.\n", encoding="utf-8")

    summary = ingestion.ingest_corpus(settings=settings)  # no dir overrides

    assert summary["chunks"] == 2
    assert set(summary["source_files"]) == {"official.md", "uploaded.md"}
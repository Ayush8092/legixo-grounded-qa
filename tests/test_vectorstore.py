"""Offline unit tests for app.vectorstore.list_all_ids / delete_vectors —
the stale-vector reconciliation primitives. Fakes stand in for the Pinecone
Index client; no network calls.
"""

from app.config import Settings
from app.vectorstore import delete_vectors, list_all_ids


def _settings(**overrides) -> Settings:
    base = dict(
        GEMINI_API_KEY="x",
        GROQ_API_KEY="x",
        PINECONE_API_KEY="x",
        PINECONE_NAMESPACE="legixo-corpus",
    )
    base.update(overrides)
    return Settings(**base)


class _FakeIndexWithList:
    """Mimics pinecone-client's `index.list(namespace=...)` generator API:
    yields batches (lists) of vector IDs."""

    def __init__(self, ids_by_namespace: dict[str, list[str]], batch_size: int = 2):
        self._ids_by_namespace = ids_by_namespace
        self._batch_size = batch_size
        self.delete_calls: list[dict] = []

    def list(self, namespace: str):
        ids = self._ids_by_namespace.get(namespace, [])
        for start in range(0, len(ids), self._batch_size):
            yield ids[start : start + self._batch_size]

    def delete(self, ids, namespace):
        self.delete_calls.append({"ids": list(ids), "namespace": namespace})


class _FakeIndexWithoutList:
    """An older/incompatible Pinecone client without `.list()` support."""

    def delete(self, ids, namespace):  # pragma: no cover - must never be called
        raise AssertionError("delete() should not be called without list() support")


def test_list_all_ids_returns_every_id_across_pagination_batches():
    settings = _settings()
    index = _FakeIndexWithList({"legixo-corpus": ["a::1::0", "a::2::0", "b::1::0", "b::2::0", "c::1::0"]})

    ids = list_all_ids(index, settings)

    assert ids == {"a::1::0", "a::2::0", "b::1::0", "b::2::0", "c::1::0"}


def test_list_all_ids_only_reads_the_configured_namespace():
    """Vectors in a different namespace must never be visible to reconciliation."""
    settings = _settings()
    index = _FakeIndexWithList(
        {
            "legixo-corpus": ["a::1::0"],
            "some-other-namespace": ["z::1::0", "z::2::0"],
        }
    )

    ids = list_all_ids(index, settings)

    assert ids == {"a::1::0"}
    assert "z::1::0" not in ids


def test_list_all_ids_degrades_to_empty_set_when_client_lacks_list_support(capsys):
    settings = _settings()
    index = _FakeIndexWithoutList()

    ids = list_all_ids(index, settings)

    assert ids == set()
    assert "WARNING" in capsys.readouterr().out


def test_list_all_ids_degrades_gracefully_on_a_listing_error(capsys):
    settings = _settings()

    class _FlakyIndex:
        def list(self, namespace):
            raise RuntimeError("transient Pinecone error")

    ids = list_all_ids(_FlakyIndex(), settings)

    assert ids == set()
    assert "WARNING" in capsys.readouterr().out


def test_delete_vectors_batches_and_scopes_to_configured_namespace():
    settings = _settings()
    index = _FakeIndexWithList({"legixo-corpus": []})
    ids_to_delete = [f"stale::{i}::0" for i in range(5)]

    deleted_count = delete_vectors(index, settings, ids_to_delete)

    assert deleted_count == 5
    all_deleted_ids = [i for call in index.delete_calls for i in call["ids"]]
    assert set(all_deleted_ids) == set(ids_to_delete)
    assert all(call["namespace"] == "legixo-corpus" for call in index.delete_calls)


def test_delete_vectors_with_empty_list_is_a_noop():
    settings = _settings()
    index = _FakeIndexWithList({"legixo-corpus": []})

    deleted_count = delete_vectors(index, settings, [])

    assert deleted_count == 0
    assert index.delete_calls == []


def test_delete_vectors_never_touches_delete_all():
    """Reconciliation must always name specific IDs — never wipe the namespace."""
    settings = _settings()
    index = _FakeIndexWithList({"legixo-corpus": []})

    delete_vectors(index, settings, ["a::1::0"])

    for call in index.delete_calls:
        assert "delete_all" not in call
        assert call["ids"] == ["a::1::0"]

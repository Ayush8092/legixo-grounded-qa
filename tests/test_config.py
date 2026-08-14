"""Offline unit tests for app.config — SecretStr handling and validation bounds."""

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def _base(**overrides) -> dict:
    base = dict(
        GEMINI_API_KEY="g",
        GROQ_API_KEY="q",
        PINECONE_API_KEY="p",
    )
    base.update(overrides)
    return base


def test_api_keys_are_secretstr_not_plain_str():
    settings = Settings(**_base())

    assert isinstance(settings.gemini_api_key, SecretStr)
    assert isinstance(settings.groq_api_key, SecretStr)
    assert isinstance(settings.pinecone_api_key, SecretStr)


def test_secretstr_repr_never_leaks_the_value():
    settings = Settings(
        **_base(GEMINI_API_KEY="super-secret-value")
    )

    assert "super-secret-value" not in repr(settings.gemini_api_key)
    assert "super-secret-value" not in str(settings.gemini_api_key)


def test_get_secret_value_returns_the_real_key():
    settings = Settings(
        **_base(GEMINI_API_KEY="super-secret-value")
    )

    assert (
        settings.gemini_api_key.get_secret_value()
        == "super-secret-value"
    )


def test_missing_required_key_raises_validation_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(
            GROQ_API_KEY="q",
            PINECONE_API_KEY="p",
            _env_file=None,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("TOP_K", 0),
        ("TOP_K", 21),
        ("SCORE_THRESHOLD", -0.1),
        ("SCORE_THRESHOLD", 1.1),
        ("ANSWER_TEMPERATURE", -0.1),
        ("ANSWER_TEMPERATURE", 1.1),
        ("API_PORT", 0),
        ("API_PORT", 70000),
        ("MAX_RETRIEVAL_LOOPS", -1),
        ("QUERY_FANOUT", 0),
        ("EMBEDDING_DIMENSION", 0),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**_base(**{field: value}))


def test_defaults_are_sensible():
    settings = Settings(**_base())

    assert settings.top_k >= 1
    assert 0.0 <= settings.score_threshold <= 1.0
    assert 0.0 <= settings.answer_temperature <= 1.0
    assert settings.max_retrieval_loops >= 0
    assert settings.recursion_limit >= settings.max_retrieval_loops


def test_get_settings_is_cached(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("PINECONE_API_KEY", "p")

    first = get_settings()
    second = get_settings()

    assert first is second

    get_settings.cache_clear()
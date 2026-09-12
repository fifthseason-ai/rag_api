import os
import pytest

from app.config import (
    RAG_HOST,
    RAG_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACT_IMAGES,
    VECTOR_DB_TYPE,
    EmbeddingsProvider,
    resolve_embeddings_provider,
    require_auth_config,
)


def test_config_defaults():
    assert RAG_HOST is not None
    assert isinstance(RAG_PORT, int)
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(PDF_EXTRACT_IMAGES, bool)
    assert VECTOR_DB_TYPE is not None


# --- EMBEDDINGS_PROVIDER fail-closed (D-KSPT-2) ----------------------------


def test_embeddings_provider_missing_raises():
    with pytest.raises(ValueError) as exc:
        resolve_embeddings_provider(None)
    # The message must list the accepted values so the operator can fix it.
    assert "EMBEDDINGS_PROVIDER" in str(exc.value)
    assert "bedrock" in str(exc.value)


def test_embeddings_provider_empty_raises():
    with pytest.raises(ValueError):
        resolve_embeddings_provider("")
    with pytest.raises(ValueError):
        resolve_embeddings_provider("   ")


def test_embeddings_provider_unknown_raises():
    with pytest.raises(ValueError) as exc:
        resolve_embeddings_provider("definitely-not-a-provider")
    assert "Unknown EMBEDDINGS_PROVIDER" in str(exc.value)


def test_embeddings_provider_valid_resolves():
    # Compare by value/name to stay robust against the enum-class identity split
    # that pytest's import machinery can introduce (the running app imports
    # app.config exactly once, so this is a test-harness artifact only).
    assert type(resolve_embeddings_provider("bedrock")).__name__ == "EmbeddingsProvider"
    assert resolve_embeddings_provider("bedrock").value == "bedrock"
    assert resolve_embeddings_provider("bedrock").name == "BEDROCK"
    assert resolve_embeddings_provider("openai").value == "openai"
    # Case/whitespace tolerant.
    assert resolve_embeddings_provider(" Bedrock ").value == "bedrock"


def test_no_default_provider_kept():
    # There must be no silent default: openai is not returned for a missing value.
    with pytest.raises(ValueError):
        resolve_embeddings_provider(None)


# --- JWT auth startup guard (D-KSPT-1) -------------------------------------


def test_require_auth_config_raises_without_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError) as exc:
        require_auth_config()
    assert "JWT_SECRET" in str(exc.value)


def test_require_auth_config_ok_with_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "s")
    require_auth_config()  # must not raise


def test_require_auth_config_has_no_bypass(monkeypatch):
    # The opt-in was removed: RAG_AUTH_DISABLED must NOT re-enable an auth-less
    # startup. Missing JWT_SECRET fails closed unconditionally.
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("RAG_AUTH_DISABLED", "true")
    assert not hasattr(__import__("app.config", fromlist=["x"]), "RAG_AUTH_DISABLED")
    with pytest.raises(RuntimeError):
        require_auth_config()

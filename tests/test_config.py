import os
import sys
import subprocess
import textwrap

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
    # Case/whitespace tolerant.
    assert resolve_embeddings_provider(" Bedrock ").value == "bedrock"


def test_no_default_provider_kept():
    # There must be no silent default: openai is not returned for a missing value.
    with pytest.raises(ValueError):
        resolve_embeddings_provider(None)


# --- Approved-provider allow-list (D-KSPT-2; ported from RATB-01 5d9fe48) ----


def test_bedrock_approved_by_default():
    # bedrock is the default approved provider.
    assert resolve_embeddings_provider("bedrock", approved=["bedrock"]).value == "bedrock"


def test_explicit_openai_with_default_allowlist_raises():
    # A syntactically-valid but UNAPPROVED provider fails closed. Default allow-list
    # is bedrock-only; openai must be rejected even though it is a real provider.
    with pytest.raises(ValueError) as exc:
        resolve_embeddings_provider("openai", approved=["bedrock"])
    assert "not in the approved" in str(exc.value)
    assert "openai" in str(exc.value)


def test_openai_accepted_when_explicitly_approved():
    # Widening the allow-list makes openai usable — proving the mechanism, not a
    # recommendation.
    assert (
        resolve_embeddings_provider("openai", approved=["bedrock", "openai"]).value
        == "openai"
    )


# --- Fail-closed startup at import (subprocess, fresh env) ------------------
# The in-process interpreter already imported a valid app.config via conftest, so
# module-level fail-closed behavior is exercised in a fresh subprocess with a fully
# controlled environment. The bootstrap patches pgvector __post_init__ so the
# "import succeeds" cases do not attempt a real DB connection.

_BOOTSTRAP = textwrap.dedent(
    """
    from langchain_community.vectorstores.pgvector import PGVector
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    PGVector.__post_init__ = lambda self: None
    AsyncPgVector.__post_init__ = lambda self: None
    import app.config  # noqa: F401
    """
)


def _import_config(env_overrides):
    """Import app.config in a subprocess with a controlled env."""
    env = dict(os.environ)
    for key in (
        "JWT_SECRET",
        "EMBEDDINGS_PROVIDER",
        "RAG_APPROVED_EMBEDDINGS_PROVIDERS",
        "OPENAI_API_KEY",
        "RAG_OPENAI_API_KEY",
    ):
        env.pop(key, None)
    # Offline dummy AWS creds so a bedrock import never needs the network.
    env["AWS_ACCESS_KEY_ID"] = "testing"
    env["AWS_SECRET_ACCESS_KEY"] = "testing"
    env["AWS_DEFAULT_REGION"] = "us-east-1"
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
    )


def test_config_missing_embeddings_provider_fails_closed():
    result = _import_config({})
    assert result.returncode != 0, result.stderr
    assert "EMBEDDINGS_PROVIDER is required" in result.stderr


def test_config_unapproved_openai_provider_fails_closed():
    # openai is a valid provider but NOT in the default approved set (bedrock).
    result = _import_config({"EMBEDDINGS_PROVIDER": "openai", "OPENAI_API_KEY": "k"})
    assert result.returncode != 0, result.stderr
    assert "not in the approved" in result.stderr
    assert "openai" in result.stderr


def test_config_approved_bedrock_provider_imports():
    result = _import_config({"EMBEDDINGS_PROVIDER": "bedrock"})
    assert result.returncode == 0, result.stderr


def test_config_openai_allowed_when_explicitly_approved():
    result = _import_config(
        {
            "EMBEDDINGS_PROVIDER": "openai",
            "RAG_APPROVED_EMBEDDINGS_PROVIDERS": "bedrock,openai",
            "OPENAI_API_KEY": "test_key",
        }
    )
    assert result.returncode == 0, result.stderr


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

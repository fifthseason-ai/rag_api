import os
import sys
import subprocess
import textwrap

from app.config import RAG_HOST, RAG_PORT, CHUNK_SIZE, CHUNK_OVERLAP, PDF_EXTRACT_IMAGES, VECTOR_DB_TYPE


def test_config_defaults():
    assert RAG_HOST is not None
    assert isinstance(RAG_PORT, int)
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(PDF_EXTRACT_IMAGES, bool)
    assert VECTOR_DB_TYPE is not None


# --- RATB-01: fail-closed startup configuration ---
#
# app.config raises at import when JWT_SECRET or an approved EMBEDDINGS_PROVIDER
# is missing/unapproved. We exercise that in a fresh subprocess with a fully
# controlled environment (the in-process interpreter has already imported a valid
# app.config via conftest, so it cannot be re-tested here). The bootstrap patches
# the pgvector __post_init__ so the "import succeeds" cases do not attempt a real
# DB connection — the import either raises before vector-store construction (the
# fail-closed cases) or succeeds offline (the approved-provider cases).

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
    """Import app.config in a subprocess with a controlled env.

    Returns the CompletedProcess (check returncode / stderr).
    """
    env = dict(os.environ)
    # Strip anything the outer test env / conftest set so each case is controlled.
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
        env=env,
        capture_output=True,
        text=True,
    )


def test_config_missing_jwt_secret_fails_closed():
    result = _import_config({"EMBEDDINGS_PROVIDER": "bedrock"})
    assert result.returncode != 0, result.stderr
    assert "JWT_SECRET is required" in result.stderr


def test_config_missing_embeddings_provider_fails_closed():
    result = _import_config({"JWT_SECRET": "s3cret"})
    assert result.returncode != 0, result.stderr
    assert "EMBEDDINGS_PROVIDER is required" in result.stderr


def test_config_unapproved_openai_provider_fails_closed():
    # openai is a valid provider but NOT in the default approved set (bedrock).
    result = _import_config(
        {"JWT_SECRET": "s3cret", "EMBEDDINGS_PROVIDER": "openai", "OPENAI_API_KEY": "k"}
    )
    assert result.returncode != 0, result.stderr
    assert "not in the approved" in result.stderr
    assert "openai" in result.stderr


def test_config_approved_bedrock_provider_imports():
    result = _import_config({"JWT_SECRET": "s3cret", "EMBEDDINGS_PROVIDER": "bedrock"})
    assert result.returncode == 0, result.stderr


def test_config_openai_allowed_when_explicitly_approved():
    # Approving openai via the allow-list makes it usable — proving the mechanism,
    # not a recommendation. OPENAI_API_KEY is required by OpenAIEmbeddings at init.
    result = _import_config(
        {
            "JWT_SECRET": "s3cret",
            "EMBEDDINGS_PROVIDER": "openai",
            "RAG_APPROVED_EMBEDDINGS_PROVIDERS": "bedrock,openai",
            "OPENAI_API_KEY": "test_key",
        }
    )
    assert result.returncode == 0, result.stderr

# tests/conftest.py
import os

# Set environment variables early so config picks up test settings.
os.environ["TESTING"] = "1"
# Set DB_HOST (and DSN) to dummy values to avoid real connection attempts.
os.environ["DB_HOST"] = "localhost"  # or any dummy value
os.environ["DSN"] = "dummy://"
# EMBEDDINGS_PROVIDER has no default anymore (D-KSPT-2, fail-closed). Tests must
# select one explicitly before app.config is imported; embeddings are mocked so
# the concrete provider is irrelevant to the assertions.
os.environ.setdefault("EMBEDDINGS_PROVIDER", "openai")
# openai is not in the default approved set (bedrock). Approve it for the in-process
# test session so app.config imports under the mocked openai provider; the
# allow-list enforcement itself is covered by the subprocess tests in test_config.py.
os.environ.setdefault("RAG_APPROVED_EMBEDDINGS_PROVIDERS", "openai,bedrock")
# The OpenAI embeddings client validates a key at construction time. Supply a
# dummy so app.config imports under the openai provider; embeddings are mocked in
# every test, so this key is never used to make a request.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

from app.services.vector_store.async_pg_vector import AsyncPgVector

# -- Patch the vector store classes to bypass DB connection --

# Do this *before* importing any app modules.
from langchain_community.vectorstores.pgvector import PGVector

def dummy_post_init(self):
    # Skip extension creation
    pass

AsyncPgVector.__post_init__ = dummy_post_init
PGVector.__post_init__ = dummy_post_init

from langchain_core.documents import Document

class DummyVectorStore:
    def get_all_ids(self) -> list[str]:
        return ["testid1", "testid2"]
    
    def get_filtered_ids(self, ids) -> list[str]:
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    def similarity_search_with_score_by_vector(self, embedding, k: int, filter: dict):
        doc = Document(
            page_content="Queried content",
            metadata={"file_id": filter.get("file_id", "testid1"), "user_id": "testuser"},
        )
        return [(doc, 0.9)]

    def add_documents(self, docs, ids):
        return ids

    async def aadd_documents(self, docs, ids):
        return ids

    async def delete(self, ids=None, collection_only: bool = False):
        return None

    # Implement the missing as_retriever() method
    def as_retriever(self):
        # Return self or wrap with a dummy retriever if needed.
        return self


# -- One writer per file_id (FILES-01 F02) -------------------------------------------
# The production lock is a Postgres advisory lock on the asyncpg pool, and there is no
# database here (DSN is a dummy). Every in-memory store double therefore runs WITHOUT
# serialization, which is stated rather than hidden: tests/utils/test_simultaneous_write.py
# installs a lock explicitly, and exercises the real advisory lock when RAG_TEST_PG_DSN
# points at an isolated Postgres.
from app.routes import document_routes as _document_routes  # noqa: E402
from app.services.file_write_lock import NoFileWriteLock  # noqa: E402

_document_routes.file_write_lock = NoFileWriteLock()


# -- PAID-CALL HAZARD: the real Bedrock rerank client is unreachable from tests ---------
# MEASURED (RV-122 on #106, 2026-09-23): `rerank()` is ON by default (`RERANK_ENABLED`
# defaults True) and several suites drive a query route WITHOUT stubbing it -- notably
# tests/test_main.py and tests/test_entitlement_routes.py, which stub nothing. A plain
# `pytest` run therefore reached the REAL Bedrock Rerank endpoint. The reviewer only
# found it because their container had the network blocked; on a host with working
# egress the call goes out. With dummy credentials it is rejected (no inference, so no
# charge), but "probably not billed" is not the standard: under the standing
# no-paid-calls hold (MASTER-PLAN ...0L.md:854, :1232) a test suite must not be able to
# call a paid provider AT ALL, and a suite that only stays free because the credentials
# happen to be wrong is one real `.env` away from spending money.
#
# The block is on `_get_client`, NOT on `_rerank_sync`, and that placement is the point:
# a test that stubs `_rerank_sync` never reaches the client, so the ~10 suites that
# legitimately exercise rerank (fallbacks, score kinds, tie order) are untouched. Only
# the path that would open a socket is closed.
#
# Failure mode is deliberately the SAME as before, minus the network: `rerank()` already
# catches Exception and falls back to the pre-rerank order, so any test that used to see
# an auth failure now sees this instead and behaves identically -- while a test that
# genuinely needs rerank output must say so by stubbing `_rerank_sync`.
import pytest  # noqa: E402

from app.services import reranker as _reranker  # noqa: E402


@pytest.fixture(autouse=True)
def _block_the_real_rerank_provider(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError(
            "hermetic tests: the real Bedrock rerank client is blocked. "
            "Stub app.services.reranker._rerank_sync if this test needs rerank output."
        )

    monkeypatch.setattr(_reranker, "_get_client", _blocked, raising=False)

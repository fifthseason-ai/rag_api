"""An empty entitlement must never retrieve a real row (FILES-01 D-ENT-EMPTY, A06).

Richard's ruling (2026-09-21): an empty string is not a valid entitlement; reject it
explicitly; it must never grant access or silently widen scope.

test_middleware.py proves the token is refused at the gate. THIS file proves the same
against the real query path on a REAL pgvector store: an empty-ent token retrieves
nothing (it is refused), a real entitlement retrieves its own row, and a different
tenant does not see it. The wildcard mutant (remove the guard) is a source control run
separately; here the positive control is a real retrieval, so a green run is not vacuous.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service via the
Candidate-B composition); absent locally it SKIPS. Offline deterministic embeddings only.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
_SECRET = "testsecret"

needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the real query path",
)


class _DetEmb(Embeddings):
    """Deterministic offline embeddings — same text always maps to the same vector."""

    def __init__(self, size=16):
        self.size = size

    def _v(self, t):
        import hashlib

        h = hashlib.sha256(t.encode()).digest()
        return [((h[i % len(h)] / 255.0) * 2 - 1) for i in range(self.size)]

    def embed_documents(self, ts):
        return [self._v(t) for t in ts]

    def embed_query(self, t):
        return self._v(t)


def _sqlalchemy_dsn():
    d = PG_DSN
    if d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _tok(entity_ids, tid="tA", uid="uA", act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid, "tid": tid, "ent": entity_ids, "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture()
def seeded(monkeypatch):
    """A real pgvector store holding: a row owned by uA/tA, and a row owned by the EMPTY
    owner (user_id="") -- the row an empty-ent token would reach if the guard were absent."""
    import psycopg2
    from app.routes import document_routes as dr
    from app.services.vector_store.factory import get_vector_store

    raw = PG_DSN.replace("postgresql+psycopg2://", "postgresql://")
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()

    # tests/conftest.py no-ops PGVector.__post_init__ for the whole session, so a store
    # built here has no engine/tables. Run the REAL initialization by hand so this test
    # exercises a genuine pgvector store, not a half-built one.
    def _real_post_init(self):
        from langchain_community.vectorstores.pgvector import _get_embedding_collection_store
        if self.create_extension:
            self.create_vector_extension()
        EmbeddingStore, CollectionStore = _get_embedding_collection_store(
            self._embedding_length, use_jsonb=self.use_jsonb
        )
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        self.create_tables_if_not_exists()
        self.create_collection()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _DetEmb(), "ent_empty", mode="sync")
    _real_post_init(store)
    store.add_documents(
        [Document(page_content="alpha owned by userA", metadata={"file_id": "fA", "user_id": "uA", "tenant_id": "tA"})],
        ids=["fA"],
    )
    store.add_documents(
        [Document(page_content="beta owned by the empty owner", metadata={"file_id": "fEmpty", "user_id": "", "tenant_id": "tA"})],
        ids=["fEmpty"],
    )

    astore = get_vector_store(_sqlalchemy_dsn(), _DetEmb(), "ent_empty", mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)
    from app.config import vector_store as _vs  # noqa: F401
    monkeypatch.setattr("app.config.vector_store", astore, raising=False)
    if getattr(__import__("main").app.state, "thread_pool", None) is None:
        __import__("main").app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    from main import app
    return TestClient(app)


@needs_pg
def test_empty_ent_token_retrieves_nothing_and_is_refused(seeded):
    client = seeded
    # /query and /query/{entity_id}: an empty-ent token is refused at the gate (403),
    # so it never reaches the empty-owned row it would otherwise match.
    r = client.post("/query", json={"query": "beta", "file_id": "fEmpty", "k": 5},
                    headers=_tok([""]))
    assert r.status_code == 403, r.text
    r2 = client.post("/query/", json={"query": "beta", "k": 5}, headers=_tok([""]))
    assert r2.status_code in (403, 404), r2.text  # empty entity id in path or refused ent


@needs_pg
def test_real_entitlement_retrieves_its_own_row(seeded):
    """Positive control: the mechanism works, so the negative above is not vacuous."""
    client = seeded
    r = client.post("/query", json={"query": "alpha", "file_id": "fA", "k": 5},
                    headers=_tok(["uA"], uid="uA"))
    assert r.status_code == 200, r.text
    hits = r.json()
    assert hits and any("alpha" in h[0]["page_content"] for h in hits), hits


@needs_pg
def test_a_different_tenant_entitlement_does_not_see_the_row(seeded):
    client = seeded
    r = client.post("/query", json={"query": "alpha", "file_id": "fA", "k": 5},
                    headers=_tok(["uB"], tid="tB", uid="uB"))
    # Authorized token, but its entity set does not include uA, so the row is filtered out.
    assert r.status_code == 200, r.text
    assert r.json() == [], r.json()

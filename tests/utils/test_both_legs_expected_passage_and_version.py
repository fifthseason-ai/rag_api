"""P06-4 item 2 — BOTH retrieval legs on the SAME (P06-2-shaped) collection return the
EXPECTED passage and the EXPECTED source VERSION, per leg and fused.

For a representative synthetic question, each leg must retrieve the expected passage of
the expected source and carry that source's identity (file_id/filename) and version
(ingest_id):
  * dense/meaning leg (dense_only)         -> expected passage + expected ingest_id
  * keyword/FTS leg (keyword_only)         -> expected passage + expected ingest_id
  * fused (RRF over both)                  -> expected passage + expected ingest_id

WHY A SEPARATE FILE (not a duplicate of entitlement_fused / hybrid_keyword_leg):
  * test_entitlement_fused proves each leg + fused return [] for an unentitled caller
    (the NEGATIVE); its positive control only checks a row is retrievable, not that the
    correct VERSION (ingest_id) rides on each leg.
  * test_hybrid_keyword_leg proves the keyword leg matches an exact term; it does not
    assert the ingest_id/version per leg on the P06-2 identity shapes.
  This file is the POSITIVE, version-carrying, per-leg acceptance on the P06-2 collection.

The superseded-version-does-not-leak mechanic (the /embed replace path) is proven in
test_version_supersession_retrieval.py; here the collection is seeded directly at the
current version, so "expected version" is asserted independently of the replace path.

GOLDEN QUESTIONS (Q01-Q36, GOLDEN-KNOWLEDGE-QUESTIONS.md): the CONTENT-specific ones
(Q01-Q03, Q05, Q07, Q09-Q11, Q13, Q16-Q23, Q26-Q28, Q30-Q33, Q35-Q36) reference named
REAL Vivaldi sources that DO NOT EXIST on this host (synthetic-only; A3/A4 blocker), so
they cannot be run here and are not reproduced. The BEHAVIOURAL questions map onto this
lane's synthetic tests: Q14/Q25/Q34 (access denied / revoked-stays-denied) ->
test_permission_change_revocation.py; Q06/Q08/Q24/Q29 (version accuracy / a later
version supersedes) -> test_version_supersession_retrieval.py; Q12 (unread != empty) ->
test_incomplete_state_survives_retrieval.py. This file covers the "retrieve the expected
passage + version on both legs" shape common to the answerable questions.

SYNTHETIC only. Needs a Postgres with pgvector; RAG_TEST_PG_DSN selects it,
RAG_TEST_PG_REQUIRED=1 turns "no DSN" into an error so this provably ran.
"""
import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import psycopg2
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the both-legs acceptance",
)

_SECRET = "test-secret-both-legs"
_COLLECTION = "both_legs_accept"

# P06-2 collection shapes.
ENT = "ent-knowledge-1"
TENANT = "tenant-vivaldi"

# Source BRIEF (the expected answer to Q_BRIEF) and a distractor XLSX source.
F_BRIEF, NAME_BRIEF, ING_BRIEF = "kn-src-brief", "astra-brief.txt", "ingest-brief-v2"
F_XLSX, NAME_XLSX, ING_XLSX = "kn-xlsx-src", "book.xlsx", "ingest-xlsx-v1"

KW_BRIEF = "astravector1"   # exact term present ONLY in the brief passage (keyword leg)
P_BRIEF = f"astra knowledge brief: the launch readiness summary {KW_BRIEF} passage"
P_XLSX = "workbook revenue by region and quarter distractor passage"

# The question. Its dense vector is made identical to the brief passage's vector so the
# dense leg ranks the brief first; the keyword leg matches via KW_BRIEF.
Q_BRIEF = "what does the astra launch readiness brief say"

_VECTORS = {
    Q_BRIEF: [1.0, 0.0, 0.0],
    P_BRIEF: [1.0, 0.0, 0.0],
    P_XLSX:  [0.0, 1.0, 0.0],
    KW_BRIEF: [0.2, 0.2, 0.9],  # a keyword-only query: dense vector far from the brief
}


class _TableEmb:
    def embed_documents(self, texts):
        return [list(_VECTORS[t]) for t in texts]

    def embed_query(self, text):
        return list(_VECTORS[text])


def _sqlalchemy_dsn():
    d = PG_DSN
    if d and d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _tok(entity_ids=(ENT,), tid=TENANT, act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "caller", "tid": tid, "ent": list(entity_ids), "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


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


@pytest.fixture()
def env(monkeypatch):
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store

    raw = _raw_dsn()
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="sync")
    _real_post_init(store)
    store.add_documents(
        [
            Document(page_content=P_BRIEF, metadata={
                "file_id": F_BRIEF, "filename": NAME_BRIEF, "user_id": ENT,
                "tenant_id": TENANT, "ingest_id": ING_BRIEF}),
            Document(page_content=P_XLSX, metadata={
                "file_id": F_XLSX, "filename": NAME_XLSX, "user_id": ENT,
                "tenant_id": TENANT, "ingest_id": ING_XLSX}),
        ],
        ids=["rBrief", "rXlsx"],
    )
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding "
            "ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_tsv "
            "ON langchain_pg_embedding USING gin (document_tsv)"
        )
        c.commit()

    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", raw, raising=True)

    astore = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)
    monkeypatch.setattr("app.config.vector_store", astore, raising=False)
    monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(dr, "RERANK_ENABLED", False)

    import main
    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    yield TestClient(main.app)

    if PSQLDatabase.pool is not None:
        PSQLDatabase.pool.terminate()
        PSQLDatabase.pool = None


def _top(resp):
    """The best hit's (page_content, ingest_id, file_id, filename)."""
    hits = resp.json()
    assert hits, "no hits"
    m = hits[0][0]["metadata"]
    return hits[0][0]["page_content"], m.get("ingest_id"), m.get("file_id"), m.get("filename")


@needs_pg
def test_dense_leg_returns_expected_passage_and_version(env, monkeypatch):
    from app.routes import document_routes as dr
    monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", False)  # dense only
    r = env.client.post(f"/query/{ENT}", json={"query": Q_BRIEF, "k": 5}, headers=_tok())
    assert r.status_code == 200, r.text
    content, ingest, fid, name = _top(r)
    assert content == P_BRIEF, content
    assert (ingest, fid, name) == (ING_BRIEF, F_BRIEF, NAME_BRIEF), (ingest, fid, name)


@needs_pg
def test_keyword_leg_returns_expected_passage_and_version(env, monkeypatch):
    """Keyword-only: the dense arm is stubbed to contribute nothing, so a hit for the
    exact term KW_BRIEF is the keyword leg's, carrying the expected version."""
    from app.routes import document_routes as dr

    async def _no_dense(*a, **k):
        return []
    monkeypatch.setattr(dr.vector_store, "asimilarity_search_with_score_by_vector", _no_dense)

    r = env.client.post(f"/query/{ENT}", json={"query": KW_BRIEF, "k": 5}, headers=_tok())
    assert r.status_code == 200, r.text
    contents = [h[0]["page_content"] for h in r.json()]
    assert P_BRIEF in contents, ("keyword leg missed the expected passage", contents)
    assert all(P_XLSX != c for c in contents), "distractor must not match the keyword"
    briefs = [h for h in r.json() if h[0]["page_content"] == P_BRIEF]
    assert briefs[0][0]["metadata"].get("ingest_id") == ING_BRIEF, briefs[0][0]["metadata"]


@needs_pg
def test_fused_returns_expected_passage_and_version(env):
    """Default hybrid (dense + keyword fused by RRF): the expected passage tops the fused
    result and carries the expected version."""
    r = env.client.post(f"/query/{ENT}", json={"query": Q_BRIEF, "k": 5}, headers=_tok())
    assert r.status_code == 200, r.text
    content, ingest, fid, name = _top(r)
    assert content == P_BRIEF, content
    assert (ingest, fid, name) == (ING_BRIEF, F_BRIEF, NAME_BRIEF), (ingest, fid, name)


@needs_pg
def test_expected_version_rides_every_leg_identically(env, monkeypatch):
    """The version (ingest_id) for the expected source is the SAME on every leg -- no leg
    drops or rewrites provenance. Pins that 'expected version' is leg-independent."""
    from app.routes import document_routes as dr

    seen = {}
    # dense
    with monkeypatch.context() as m:
        m.setattr(dr, "HYBRID_SEARCH_ENABLED", False)
        seen["dense"] = _top(env.client.post(f"/query/{ENT}", json={"query": Q_BRIEF, "k": 5},
                                             headers=_tok()))[1]
    # fused
    seen["fused"] = _top(env.client.post(f"/query/{ENT}", json={"query": Q_BRIEF, "k": 5},
                                         headers=_tok()))[1]
    assert set(seen.values()) == {ING_BRIEF}, seen

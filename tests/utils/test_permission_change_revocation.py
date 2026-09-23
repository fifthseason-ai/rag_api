"""P06-4 item 1 — PERMISSION CHANGE: grant -> revoke -> the next query is refused
(revocation-equality), on a REAL pgvector store, on every query route.

WHAT IS GENUINELY NEW HERE (not covered by test_entitlement_fused / test_empty_
entitlement_query_path / test_entitlement_routes)
--------------------------------------------------------------------------------
The existing suites prove a *statically* empty / foreign / cross-tenant entitlement
gets [] or 403. NONE of them exercise a *transition*: a caller who WAS entitled to an
entity (and really retrieved its rows), then is revoked, and whose very next query on
the SAME collection and the SAME query text must return nothing -- no content, no
restricted metadata, no count, and no cached hit warmed by the earlier authorized call.

ARCHITECTURAL GROUND TRUTH (measured, app/middleware.py:116-175, app/routes/
document_routes.py:123-147,1063-1064,1251-1273): rag_api holds NO server-side
entitlement state and NO per-caller result cache. Authority is re-derived from the
signed token on EVERY request (`request.state.entitlement` <- `ent`/`act`/`tid`
claims). The ONLY query-path cache is `get_cached_query_embedding`, which caches the
query EMBEDDING VECTOR (keyed on query text + embeddings-model namespace) -- an
entitlement-INDEPENDENT value -- after which the entitlement filter runs fresh. So
"revocation" in rag_api == a token that no longer carries the entity, and
revocation-equality is the property that the earlier authorized call leaves NO residue.

RED-FIRST / what would make this pass WRONGLY, and whether the test catches it
--------------------------------------------------------------------------------
  * A result cache keyed on query text (not on entitlement): the revoked call would be
    served the owner's warmed rows. CAUGHT by test_revoked_after_owner_warmed_the_cache
    (owner queries first to warm; revoked call must still be [] and byte-identical to a
    cold revoked call).
  * The entity-membership check dropped from `_require_entity`: /query/{entity_id} would
    return 200 with the owner's rows instead of 403. CAUGHT by
    test_grant_then_revoke_refused (route query_by_entity asserts 403).
  * `_authorized_only` / the arm user_id filter dropped: the file-id and multiple routes
    would leak. CAUGHT here AND already by test_entitlement_fused's controls.
Positive control (test_grant_actually_returned_the_row) proves the grant really
retrieved the row first, so the revoke negative is not vacuous.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service);
RAG_TEST_PG_REQUIRED=1 turns "no DSN" from a skip into an error so this provably ran.
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
    reason="RAG_TEST_PG_DSN not set: no pgvector for the revocation query path",
)

_SECRET = "test-secret-revocation"
_COLLECTION = "perm_change_revocation"

# Built on the P06-2 collection shapes (P06-2-INGESTION-RECEIPT-2026-09-22T221458Z.md):
# identity/permission fields are the same namespace P06-2 proved survive into stored rows
# (file_id kn-*, user_id = resolved entity, tenant_id = tenant-vivaldi, filename on rows).
TENANT = "tenant-vivaldi"
ENT_X = "ent-knowledge-1"     # the entity the caller is granted then revoked from
ENT_OTHER = "ent-knowledge-2" # a valid, non-empty entitlement that never included ENT_X
FILE_X = "kn-src-brief"
FILENAME_X = "astra-brief.txt"

# One row owned by ENT_X; a rare term so both the dense and keyword arms can find it.
TERM = "revocateron9"
OWN_X = f"{TERM} {TERM} confidential knowledge remittance note"
NOTHING = "phrase that matches no row at all"

_VECTORS = {
    TERM: [1.0, 0.0, 0.0],
    NOTHING: [0.0, 0.0, 1.0],
    OWN_X: [1.0, 0.0, 0.0],
}

# Row values that identify the owner's restricted content and must never appear in a
# revoked/denied response body.
_RESTRICTED_VALUES = (OWN_X, ENT_X, FILE_X, FILENAME_X)


class _TableEmb:
    """Fixed text -> vector table. Unknown text raises: fixture drift must be loud."""

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


def _tok(entity_ids, caller="caller", tid=TENANT, act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": caller, "tid": tid, "ent": list(entity_ids), "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _real_post_init(self):
    """Run the REAL pgvector init that conftest no-ops for the session."""
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


class _Env:
    def __init__(self, client, embed_calls):
        self.client = client
        self.embed_calls = embed_calls


@pytest.fixture()
def env(monkeypatch):
    """Real pgvector holding one uX-owned row; both arms pointed at it; the query
    embedding function counted so cache-warm order can be reasoned about."""
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
        [Document(page_content=OWN_X,
                  metadata={"file_id": FILE_X, "user_id": ENT_X, "tenant_id": TENANT,
                            "filename": FILENAME_X, "ingest_id": "ingest-vX"})],
        ids=["rX"],
    )
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding "
            "ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
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

    # Count query-embedding calls: the revoked answer must not depend on whether the
    # owner warmed the (embedding) cache -- and rag_api must NOT be caching results here.
    embed_calls = []
    real_embed = astore.embedding_function.embed_query

    def _counting_embed(text):
        embed_calls.append(text)
        return real_embed(text)
    monkeypatch.setattr(astore.embedding_function, "embed_query", _counting_embed)

    # TestClient runs each request on a FRESH event loop, so an asyncpg keyword pool bound
    # to a previous request's loop is stale and must not be reused. Mirror F-ENTITLEMENT-
    # FUSED: close the pool after every keyword call so each request rebuilds it on its own
    # loop. Without this the pool errors and leaks connections that deadlock the next file's
    # DROP TABLE (measured 2026-09-23).
    real_kw = dr.keyword_search

    async def _kw(*a, **k):
        try:
            return await real_kw(*a, **k)
        finally:
            await PSQLDatabase.close_pool()
    monkeypatch.setattr(dr, "keyword_search", _kw)

    import main
    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    yield _Env(TestClient(main.app), embed_calls)

    if PSQLDatabase.pool is not None:
        PSQLDatabase.pool.terminate()
        PSQLDatabase.pool = None


# Each route as (callable(client, query, ent_claims) -> response). The revoked/never
# tokens still NAME uX where a route takes an id: the id is only a filter.
_ROUTES = {
    "query_by_entity": lambda c, q, ent: c.post(
        f"/query/{ENT_X}", json={"query": q, "k": 5}, headers=_tok(ent)),
    "query_by_file": lambda c, q, ent: c.post(
        "/query", json={"query": q, "file_id": FILE_X, "k": 5}, headers=_tok(ent)),
    "query_multiple": lambda c, q, ent: c.post(
        "/query_multiple", json={"query": q, "file_ids": [FILE_X], "k": 5}, headers=_tok(ent)),
}


def _texts(resp):
    return [hit[0]["page_content"] for hit in resp.json()]


def _body_str(resp):
    return resp.content.decode("utf-8", "ignore")


@needs_pg
def test_grant_actually_returned_the_row(env):
    """Positive control: while granted (ent=[uX]) the caller retrieves uX's row on the
    entity route -- so the revoke negatives below are not vacuous."""
    r = env.client.post(f"/query/{ENT_X}", json={"query": TERM, "k": 5},
                        headers=_tok([ENT_X]))
    assert r.status_code == 200, r.text
    assert _texts(r) == [OWN_X], _texts(r)


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_grant_then_revoke_refused(env, route):
    """GRANT then REVOKE on the same collection + query text: the granted call returns
    uX's row; the revoked call (ent=[uY], no longer includes uX) returns nothing and
    leaks no restricted field. The entity route refuses at the gate (403); the file /
    multiple routes filter to 200 []."""
    granted = _ROUTES[route](env.client, TERM, [ENT_X])
    assert granted.status_code == 200, granted.text

    revoked = _ROUTES[route](env.client, TERM, [ENT_OTHER])
    if route == "query_by_entity":
        # uX no longer in the entitlement -> _require_entity refuses.
        assert revoked.status_code == 403, revoked.text
    else:
        assert revoked.status_code == 200, revoked.text
        assert revoked.json() == [], revoked.json()

    # No restricted VALUE of the owner's row anywhere in the revoked body -- not the
    # passage, the entity id, the file id, or the filename. Assert on the values that
    # identify the row, not the key names (an empty [] legitimately contains no keys).
    body = _body_str(revoked)
    for value in _RESTRICTED_VALUES:
        assert value not in body, ("revoked body disclosed a restricted value: %r" % value, body)


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_never_entitled_denial_is_deterministic_and_leaks_no_existence_oracle(env, route):
    """NEVER-ENTITLED existence-oracle guard (NOT a grant->revoke transition: both compared
    calls carry ENT_OTHER, which never included uX, so no transition is built here). Two
    never-entitled denials are byte-for-byte identical -- same status, same body, same
    content-length -- and identical to a query that matches nothing at all, so a denied
    caller learns nothing about whether uX's row exists. The transition property is proven
    by test_grant_then_revoke_refused and test_revoked_after_owner_warmed_the_cache."""
    revoked = _ROUTES[route](env.client, TERM, [ENT_OTHER])
    never = _ROUTES[route](env.client, TERM, [ENT_OTHER])
    assert (revoked.status_code, revoked.content) == (never.status_code, never.content)
    # This header check degrades to None == None when content-length is absent; the
    # load-bearing guard is the .content byte-equality on the line above, which cannot pass
    # vacuously. Kept as a cheap corroborating signal, not as the primary assertion.
    assert revoked.headers.get("content-length") == never.headers.get("content-length")

    # And identical to a query that matches nothing at all (no count / existence leak).
    miss = _ROUTES[route](env.client, NOTHING, [ENT_OTHER])
    assert (miss.status_code, miss.content) == (revoked.status_code, revoked.content)


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_revoked_after_owner_warmed_the_cache(env, route):
    """Cache-warm-order independence: the owner queries the SAME text first (warming the
    query-embedding cache); the revoked answer is then byte-identical to the revoked
    answer with a cold cache. A result-cache keyed on query text would redden this."""
    # Cold: revoked first.
    cold = _ROUTES[route](env.client, TERM, [ENT_OTHER])

    # Warm: owner runs the same query text (retrieves the row, warms embedding cache).
    warm_owner = env.client.post(f"/query/{ENT_X}", json={"query": TERM, "k": 5},
                                 headers=_tok([ENT_X]))
    assert warm_owner.status_code == 200 and _texts(warm_owner) == [OWN_X], warm_owner.text

    # Revoked again, cache now warm.
    warm = _ROUTES[route](env.client, TERM, [ENT_OTHER])

    assert (cold.status_code, cold.content) == (warm.status_code, warm.content), \
        "the owner warming the cache changed the revoked answer"
    assert OWN_X not in _body_str(warm), _body_str(warm)


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_caller_with_a_foreign_entity_set_is_excluded_on_every_route(env, route):
    """A caller whose token carries a DIFFERENT entity set (here also a different tid, but the
    tid is NOT what excludes the row) gets nothing on every route and no restricted value
    leaks. This is the same entity-exclusion as test_grant_then_revoke_refused, exercised
    under a foreign entity id -- it is NOT a tenant-isolation test, because (per the nuance
    below) rag_api has no tenant retrieval filter to exercise.

    MEASURED NUANCE (P06-5 §2a): rag_api has no tenant retrieval filter -- `tenant_id` is
    stored on cmetadata but retrieval scopes on `user_id` (entity). Cross-tenant isolation
    therefore rests on the token minter scoping `ent` to the caller's own tenant (Core mints
    ent=[req.user.id]); a cross-tenant caller carries DIFFERENT entity ids, so the entity
    filter + `_authorized_only` exclude the row. This models that: a foreign-tenant token
    whose entity set does not include ENT_X sees nothing."""
    foreign = _tok(["ent-other-tenant-9"], caller="other", tid="tenant-other")
    if route == "query_by_entity":
        # naming ENT_X in the path while not entitled to it -> refused at the gate
        r = env.client.post(f"/query/{ENT_X}", json={"query": TERM, "k": 5}, headers=foreign)
        assert r.status_code == 403, r.text
    elif route == "query_by_file":
        r = env.client.post("/query", json={"query": TERM, "file_id": FILE_X, "k": 5},
                            headers=foreign)
        assert r.status_code == 200 and r.json() == [], r.text
    else:
        r = env.client.post("/query_multiple", json={"query": TERM, "file_ids": [FILE_X], "k": 5},
                            headers=foreign)
        assert r.status_code == 200 and r.json() == [], r.text
    body = _body_str(r)
    for value in _RESTRICTED_VALUES:
        assert value not in body, ("cross-tenant body disclosed a restricted value: %r" % value, body)


@needs_pg
def test_revocation_holds_on_the_keyword_leg_alone(env, monkeypatch):
    """The revoke must hold on EACH retrieval leg independently: with the dense arm
    stubbed to contribute nothing, the keyword (FTS) arm alone must still return nothing
    for the revoked caller -- the entitlement filter constrains the keyword arm too."""
    from app.routes import document_routes as dr

    async def _no_dense(*a, **k):
        return []
    monkeypatch.setattr(dr.vector_store, "asimilarity_search_with_score_by_vector", _no_dense)

    # Owner still finds it via the keyword arm (precondition), revoked does not.
    owner = env.client.post(f"/query/{ENT_X}", json={"query": TERM, "k": 5},
                            headers=_tok([ENT_X]))
    assert owner.status_code == 200 and _texts(owner) == [OWN_X], owner.text

    revoked = env.client.post("/query", json={"query": TERM, "file_id": FILE_X, "k": 5},
                              headers=_tok([ENT_OTHER]))
    assert revoked.status_code == 200 and revoked.json() == [], revoked.text

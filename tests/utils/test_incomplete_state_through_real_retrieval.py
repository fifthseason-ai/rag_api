"""P06-4 residual (b): the REAL retrieval path invents no completeness claim, and honest
per-chunk provenance survives it unchanged.

THE GAP THIS CLOSES, NAMED BY THE FILE THAT HAS IT
--------------------------------------------------
`test_incomplete_state_survives_retrieval.py` (PR #88) says so itself, in its own LIMITATION
section: its `_client` fixture monkeypatches `document_routes._retrieve_documents` with a stub
that echoes FakeStore rows, so "a completeness field fabricated INSIDE the real retrieval path
would therefore be invisible to them", and "a pg-backed variant that hits the REAL retrieval
path is NOT present in this file".

That is the whole reason this file exists. A test cannot observe a fabrication inside the path
it stubs. So here nothing on the retrieval path is stubbed: rows go into real pgvector and come
back through the real `/query` route, the real hybrid/dense search and the real response model.

WHAT IS UNDER TEST, AND WHAT IS NOT
-----------------------------------
NOT re-proven here -- REACHABLE FROM THIS TREE, so a reader can check it:
  * the /embed receipt reports partial/unverified as such and never promotes to indexed
    -- tests/utils/test_parse_is_not_index.py, present on the base.

PROVEN HERE RATHER THAN CITED. The producer-side half -- that `_prepare_documents_sync`
never stamps a completeness field onto a stored chunk -- was originally cited to
`test_incomplete_state_survives_retrieval`, which lives ONLY on the unmerged
search-perms branch and is absent from this change and from the base. **A citation is a
claim about the tree the reader has, not about any tree anywhere**, so it read as
corroboration to anyone who did not go looking, with nothing behind it for them. It is
asserted directly below instead (`test_the_producer_stamps_no_completeness_field`), which
removes the dependency rather than annotating it.

For the record, since the distinction cost a review round: that sibling file is NOT
unrun. Its commit TITLE says "(slot-free prep, UNRUN)" and the title is stale -- the file
was executed in the P06-4 slot run at 2026-09-23T02:15Z, 3 passed. A commit title is a
claim true when written; it does not update itself when the state it describes changes.
The citation was still wrong, for the reachability reason above and not for that one.

UNDER TEST HERE: everything between the store and the caller. `index.status` is a WRITE-TIME
receipt property returned on the /embed body (P06-5 §3a); it is NOT written onto per-chunk
`cmetadata`. So a retrieval-only consumer must see exactly what was stored and nothing more --
if anything on the retrieval path invented a `status` or `indexed` key, a partial source would
start looking complete to every consumer that never read the receipt.

WHY SET EQUALITY RATHER THAN A FORBIDDEN-NAME LIST
---------------------------------------------------
`test_the_retrieved_metadata_is_exactly_what_was_stored` asserts the returned key set EQUALS the
stored key set. A forbidden-name list only catches the names I happened to think of, and the
fabricated key that matters is the one nobody predicted -- `ingest_state`, `verified`, `ready`.
Set equality catches any invention whatever it is called. The named-key test below it is kept
only because it fails readably, not because it adds coverage.

NON-VACUITY
-----------
`test_honest_provenance_survives_the_real_retrieval_path` is the positive control. "The response
carries no completeness field" is satisfied perfectly by a response carrying no metadata at all,
by a store that returned nothing, or by a query that matched nothing -- so a green here means
nothing unless the honest keys demonstrably make the same trip.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it; RAG_TEST_PG_REQUIRED=1 turns "no
DSN" from a skip into an error, so the counts prove these ran.
"""
import datetime
import os

import jwt
import psycopg2
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")

needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the real-retrieval path",
)

_SECRET = "test-secret-incomplete-real"
_COLLECTION = "incomplete_real_retrieval"

ENTITY = "ent-incs"
FILE_ID = "file-partially-ingested"
TEXT = "quarterly figures for the northern region"
QUERY = TEXT

#: Everything the real ingest path stamps that a retrieval consumer legitimately reads. The
#: point of the positive control is that ALL of this survives; the point of the negative is
#: that NOTHING ELSE appears.
STORED_METADATA = {
    "file_id": FILE_ID,
    "user_id": ENTITY,
    "digest": "d41d8cd98f00b204e9800998ecf8427e",
    "document_origin_type": "upload",
    "tenant_id": "tenant-incs",
    "ingest_id": "ing-0001",
    "text_source": "ocr",
    "ocr_confidence": 0.71,
    "page": 3,
}

#: Names a fabricated completeness claim would plausibly take. Kept for a readable failure,
#: NOT relied on for coverage -- set equality above is what actually catches an invention.
FORBIDDEN = {
    "status", "index_status", "indexed", "index", "complete", "completeness",
    "is_indexed", "ingest_status", "verified", "ready", "state",
}

_VECTORS = {TEXT: [1.0, 0.0, 0.0], QUERY: [1.0, 0.0, 0.0]}


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


def _tok():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "caller", "tid": "tenant-incs", "ent": [ENTITY], "act": ["read", "write"],
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
    """Real pgvector, real routes, and NOTHING on the retrieval path stubbed.

    `_retrieve_documents` is deliberately left alone -- stubbing it is exactly what made the
    #88 suite unable to see a fabrication inside it.
    """
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store
    from main import app

    raw = _raw_dsn()
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE"
        )
        c.commit()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="sync")
    _real_post_init(store)
    store.add_documents(
        [Document(page_content=TEXT, metadata=dict(STORED_METADATA))],
        ids=[FILE_ID],
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

    # NOT `with TestClient(app)`. Entering the context runs the app lifespan, and
    # LEAVING it calls app.state.thread_pool.shutdown(wait=True) on the module-level app
    # shared by the whole test session -- every later test that needs the pool then dies
    # with "cannot schedule new futures after shutdown". Measured on the sibling suite:
    # 158 failed / 22 errors in a full run with a real DB, while every per-file run
    # stayed green, because the poisoning only reaches tests that run AFTER it in the
    # same process. A per-file green cannot see this class of defect at all.
    #
    # Same construction as test_entitlement_fused and test_ids_entitlement_scope.
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    yield TestClient(app)


def _one_hit(env):
    r = env.post(
        "/query", json={"query": QUERY, "file_id": FILE_ID, "k": 5}, headers=_tok()
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body, (
        "the real retrieval path returned NO hit, so every negative in this file would pass "
        "for the wrong reason. Response: %r" % (body,))
    document, _score = body[0]
    return document["metadata"]


# ---------------------------------------------------------------------------
# POSITIVE CONTROL — read first
# ---------------------------------------------------------------------------


@needs_pg
def test_honest_provenance_survives_the_real_retrieval_path(env):
    """NON-VACUITY GUARD. Everything else here asserts an ABSENCE, and an absence is satisfied
    by an empty response. This proves the metadata actually makes the trip."""
    md = _one_hit(env)

    for key in ("text_source", "ocr_confidence", "ingest_id", "page", "file_id"):
        assert key in md, (
            "%r did not survive the real retrieval path (got %r). The absence assertions in "
            "this file are vacuous until this passes." % (key, sorted(md)))
    assert md["text_source"] == "ocr"
    assert md["ingest_id"] == "ing-0001"
    assert md["page"] == 3


# ---------------------------------------------------------------------------
# THE PROPERTY — nothing is invented between the store and the caller
# ---------------------------------------------------------------------------


@needs_pg
def test_the_retrieved_metadata_is_exactly_what_was_stored(env):
    """THE LOAD-BEARING ASSERTION. Set equality catches ANY invented key, including ones no
    forbidden-name list would have predicted."""
    md = _one_hit(env)

    invented = set(md) - set(STORED_METADATA)
    dropped = set(STORED_METADATA) - set(md)
    assert not invented, (
        "the retrieval path INVENTED metadata the store never held: %r. A retrieval-only "
        "consumer reads this as a property of the source." % sorted(invented))
    assert not dropped, (
        "the retrieval path DROPPED stored metadata: %r. Losing provenance silently is the "
        "other half of the same defect." % sorted(dropped))


@needs_pg
def test_no_completeness_claim_appears_anywhere_in_the_hit(env):
    """Kept for a readable failure, not for coverage — set equality above is the real guard.

    Checked against the whole serialized hit, not just the metadata dict, because a claim
    added at the response-model level would not be inside `metadata` at all.
    """
    r = env.post(
        "/query", json={"query": QUERY, "file_id": FILE_ID, "k": 5}, headers=_tok()
    )
    assert r.status_code == 200, r.text
    document, _score = r.json()[0]

    present = FORBIDDEN & set(document.get("metadata", {}))
    assert not present, (
        "a completeness claim reached the caller in metadata: %r. `index.status` is a "
        "WRITE-TIME receipt property (P06-5 §3a) and must never appear per-chunk: a partial "
        "source would look complete to every consumer that did not read the receipt."
        % sorted(present))
    assert FORBIDDEN & set(document) == set(), (
        "a completeness claim reached the caller at the TOP LEVEL of the hit: %r"
        % sorted(FORBIDDEN & set(document)))


@needs_pg
def test_a_partial_source_is_not_described_at_all_rather_than_described_as_complete(env):
    """The honest shape of the contract: chunks carry no index state in EITHER direction.

    This is not the same as "the chunk says partial". It says nothing, and P06-5 §3a tells
    Core the only place completeness lives is the /embed receipt. A test asserting the chunk
    reports `partial` would be pinning a field that must not exist.
    """
    md = _one_hit(env)
    assert "index" not in md and "status" not in md
    assert md == STORED_METADATA, (
        "the stored metadata and the retrieved metadata differ: stored %r, retrieved %r"
        % (STORED_METADATA, md))


# ---------------------------------------------------------------------------
# THE PRODUCER HALF — asserted here so this file does not depend on an unmerged branch
# ---------------------------------------------------------------------------


def test_the_producer_stamps_no_completeness_field():
    """`_prepare_documents_sync` is where chunk metadata is built. It must not invent an
    index-state key, because `index.status` is a WRITE-TIME receipt property (P06-5 §3a)
    and a per-chunk copy would make a partial source look complete to every consumer that
    never read the receipt.

    No database and no route: this is the producer called directly, which is the only
    place the stamping decision is made. Set equality against the caller-supplied keys
    would be wrong here -- the producer legitimately ADDS file_id, digest and friends --
    so this asserts the forbidden set specifically, and the retrieval-side test above
    carries the stronger any-invention guard.
    """
    from app.routes.document_routes import _prepare_documents_sync

    prepared = _prepare_documents_sync(
        [Document(page_content=TEXT, metadata={"page": 3})],
        file_id=FILE_ID,
        user_id=ENTITY,
        clean_content=False,
        filename="quarterly.pdf",
        tenant_id="tenant-incs",
        ingest_id="ing-0001",
    )

    assert prepared, "the producer returned no chunks, so this asserts nothing"
    for chunk in prepared:
        present = FORBIDDEN & set(chunk.metadata)
        assert not present, (
            "the producer stamped a completeness claim onto a stored chunk: %r. "
            "metadata=%r" % (sorted(present), sorted(chunk.metadata)))

    # POSITIVE CONTROL: the producer really did stamp, so "no forbidden key" is not
    # satisfied by a chunk carrying no metadata at all.
    assert prepared[0].metadata.get("file_id") == FILE_ID, prepared[0].metadata
    assert prepared[0].metadata.get("page") == 3, (
        "the loader's own key did not survive the producer: %r" % prepared[0].metadata)

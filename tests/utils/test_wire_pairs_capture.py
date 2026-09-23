"""Cut the REAL request/response pairs for CORE's P06 citation capture.

Not transcribed from test source. Every pair below is produced by issuing the request
against the real routes over a real pgvector store and printing the response verbatim,
because a pair typed from memory is exactly the artefact that looks like evidence and is
not.

CORE's requested shape: the smallest reproducible request/response per denial class, so
the receipt can show the wire beside the screen.

PARTITION (agreed with CORE):
  0. POSITIVE CONTROL -- an entitled caller succeeds. Without it a denial proves nothing,
     because a broken pipeline denies everyone. Printed FIRST for that reason.
  1. denied same-tenant caller, no grant
  2. cross-tenant caller holding a valid NON-EMPTY entitlement
  3. cross-tenant caller at the METADATA exits
  4. NO-ORACLE pairs -- a foreign id answered byte-identically to one that never existed
  5. malformed / empty entitlement, refused by middleware before any route runs

Run inside the test container with RAG_TEST_PG_DSN pointing at a live pgvector.
"""
import datetime
import json
import os

import pytest
import jwt
import psycopg2
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from langchain_core.documents import Document

SECRET = "wire-pairs-secret"
COLLECTION = "wire_pairs"
PG_DSN = os.environ.get("RAG_TEST_PG_DSN")  # resolved lazily: an eager read would ERROR
# the whole suite's COLLECTION for anyone without a DSN, which is a far worse defect
# than this file failing to run.

TENANT_A, TENANT_B = "tenant-alpha", "tenant-beta"
ENT_A, ENT_B, ENT_NOGRANT = "ent-alpha", "ent-beta", "ent-nogrant"
FILE_A, FILE_B = "file-alpha-confidential", "file-beta-routine"
ABSENT_FILE = "file-that-was-never-stored"
ABSENT_ENT = "ent-that-never-existed"

MARKER = "vorthaxil9"
A_TEXT = f"{MARKER} {MARKER} acme merger 2026 confidential board memo"
B_TEXT = "routine tenant B expense note"
VECTORS = {MARKER: [1.0, 0.0, 0.0], A_TEXT: [1.0, 0.0, 0.0], B_TEXT: [0.0, 1.0, 0.0]}


class Emb:
    def embed_documents(self, texts):
        return [list(VECTORS[t]) for t in texts]

    def embed_query(self, text):
        return list(VECTORS[text])


def tok(tenant, ents, act=("read", "write")):
    os.environ["JWT_SECRET"] = SECRET
    payload = {
        "id": f"caller-of-{tenant}", "tid": tenant, "ent": list(ents),
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, SECRET, algorithm="HS256")}


def claims(headers):
    """The token's CLAIMS, not the token. A receipt that prints a bearer token is a
    receipt nobody can safely attach to a card."""
    raw = headers["Authorization"].split(" ", 1)[1]
    c = jwt.decode(raw, SECRET, algorithms=["HS256"])
    return {k: c[k] for k in ("id", "tid", "ent", "act") if k in c}


def real_post_init(self):
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store
    if self.create_extension:
        self.create_vector_extension()
    E, C = _get_embedding_collection_store(self._embedding_length, use_jsonb=self.use_jsonb)
    self.CollectionStore, self.EmbeddingStore = C, E
    self.create_tables_if_not_exists()
    self.create_collection()


def setup():
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store
    from main import app

    raw = PG_DSN.replace("postgresql+psycopg2://", "postgresql://")
    sa = raw.replace("postgresql://", "postgresql+psycopg2://", 1)
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()

    os.environ["JWT_SECRET"] = SECRET
    store = get_vector_store(sa, Emb(), COLLECTION, mode="sync")
    real_post_init(store)
    store.add_documents(
        [
            Document(page_content=A_TEXT, metadata={"file_id": FILE_A, "user_id": ENT_A, "tenant_id": TENANT_A}),
            Document(page_content=B_TEXT, metadata={"file_id": FILE_B, "user_id": ENT_B, "tenant_id": TENANT_B}),
        ],
        ids=[FILE_A, FILE_B],
    )
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding ADD COLUMN IF NOT EXISTS document_tsv "
            "tsvector GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        c.commit()

    PSQLDatabase.pool = None
    db.DSN = raw
    astore = get_vector_store(sa, Emb(), COLLECTION, mode="async")
    real_post_init(astore)
    dr.vector_store = astore
    dr.HYBRID_SEARCH_ENABLED = True
    dr.RERANK_ENABLED = False
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


def pair(client, label, method, path, headers, **kw):
    fn = getattr(client, method.lower())
    r = fn(path, headers=headers, **kw)
    body = kw.get("json")
    print("-" * 78)
    print(label)
    print("  REQUEST   %s %s" % (method, path))
    print("  CLAIMS    %s" % json.dumps(claims(headers), sort_keys=True))
    if body is not None:
        print("  BODY      %s" % json.dumps(body, sort_keys=True))
    print("  STATUS    %d" % r.status_code)
    print("  RESPONSE  %s" % r.text.strip())
    return r.status_code, r.text.strip()


@pytest.mark.skipif(
    not os.environ.get("CUT_WIRE_PAIRS"),
    reason="receipt generator, not a test: set CUT_WIRE_PAIRS=1 to cut the pairs",
)
def test_cut_wire_pairs(capsys):
    client = setup()
    B = tok(TENANT_B, [ENT_B])
    NOGRANT = tok(TENANT_A, [ENT_NOGRANT])

    print("=" * 78)
    print("P06 CITATION DENIAL WIRE PAIRS -- rag_api, real pgvector, real routes")
    print("Both tenants' rows live in ONE collection: separate collections would pass")
    print("through physical separation production does not have.")
    print("=" * 78)

    print("\n### CLASS 0 -- POSITIVE CONTROL (read this first)")
    print("A denial captured without a working positive control proves nothing, because a")
    print("broken pipeline denies everyone.")
    pair(client, "0a  entitled caller retrieves its OWN passage", "POST", "/query",
         B, json={"query": B_TEXT, "file_id": FILE_B, "k": 5})

    print("\n### CLASS 1 -- denied same-tenant caller (valid token, entity with no rows)")
    for p, body in (
        ("/query", {"query": MARKER, "file_id": FILE_A, "k": 5}),
        ("/query_multiple", {"query": MARKER, "file_ids": [FILE_A], "k": 5}),
    ):
        pair(client, "1  same tenant, no grant -> %s" % p, "POST", p, NOGRANT, json=body)
    pair(client, "1  same tenant, no grant -> /query/{entity_id}", "POST",
         "/query/%s" % ENT_A, NOGRANT, json={"query": MARKER, "k": 5})

    print("\n### CLASS 2 -- CROSS-TENANT caller with a valid NON-EMPTY entitlement")
    pair(client, "2  cross-tenant -> /query naming the other tenant's file", "POST",
         "/query", B, json={"query": MARKER, "file_id": FILE_A, "k": 5})
    pair(client, "2  cross-tenant -> /query_multiple", "POST", "/query_multiple",
         B, json={"query": MARKER, "file_ids": [FILE_A], "k": 5})
    pair(client, "2  cross-tenant -> /query/{entity_id} of the other tenant", "POST",
         "/query/%s" % ENT_A, B, json={"query": MARKER, "k": 5})

    print("\n### CLASS 3 -- CROSS-TENANT at the METADATA exits")
    pair(client, "3  cross-tenant -> GET /ids", "GET", "/ids", B)
    pair(client, "3  cross-tenant -> GET /documents?ids=<foreign>", "GET",
         "/documents?ids=%s" % FILE_A, B)
    pair(client, "3  cross-tenant -> GET /documents/{foreign}/context", "GET",
         "/documents/%s/context" % FILE_A, B)

    print("\n### CLASS 4 -- NO-ORACLE PAIRS  (CORE: show these side by side on screen)")
    print("A foreign id must be answered BYTE-IDENTICALLY to one that never existed.")
    print("If the two differ, the difference IS the disclosure: it confirms the file is real.")
    checks = []
    a = pair(client, "4a FOREIGN file id -> /documents", "GET",
             "/documents?ids=%s" % FILE_A, B)
    b = pair(client, "4a ABSENT  file id -> /documents", "GET",
             "/documents?ids=%s" % ABSENT_FILE, B)
    checks.append(("/documents foreign == absent", a, b))
    a = pair(client, "4b FOREIGN file id -> /context", "GET",
             "/documents/%s/context" % FILE_A, B)
    b = pair(client, "4b ABSENT  file id -> /context", "GET",
             "/documents/%s/context" % ABSENT_FILE, B)
    checks.append(("/context foreign == absent", a, b))
    a = pair(client, "4c FOREIGN entity -> /query/{entity_id}", "POST",
             "/query/%s" % ENT_A, B, json={"query": MARKER, "k": 5})
    b = pair(client, "4c ABSENT  entity -> /query/{entity_id}", "POST",
             "/query/%s" % ABSENT_ENT, B, json={"query": MARKER, "k": 5})
    checks.append(("/query/{entity} foreign == absent", a, b))

    print("\n### CLASS 5 -- malformed entitlement, refused BEFORE any route runs")
    pair(client, "5  empty-string entitlement entry", "POST", "/query",
         tok(TENANT_B, [""]), json={"query": MARKER, "file_id": FILE_A, "k": 5})
    pair(client, "5  empty entitlement list", "POST", "/query",
         tok(TENANT_B, []), json={"query": MARKER, "file_id": FILE_A, "k": 5})

    print("\n" + "=" * 78)
    print("NO-ORACLE VERDICTS (the machine-checkable part of this receipt)")
    ok = True
    for name, x, y in checks:
        same = x == y
        ok = ok and same
        print("  %-38s %s   (%s vs %s)" % (name, "IDENTICAL" if same else "*** DIFFERENT ***",
                                           x[0], y[0]))
    print("=" * 78)
    assert ok, "a no-oracle pair DIFFERED: the difference is itself the disclosure"

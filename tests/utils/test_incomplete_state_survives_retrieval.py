"""P06-4 item 4 — truthful incomplete-processing states are preserved, and the query
path NEVER promotes a source to "complete"/"indexed".

MEASURED CONTRACT FACT (this file documents it as an executable assertion)
--------------------------------------------------------------------------------
`index.status` (indexed / partial / unverified) is a WRITE-TIME RECEIPT property: it is
computed by `_index_receipt` reading the store back after insert and is returned on the
`/embed` body (P06-5 §3a). It is NOT written onto per-chunk `cmetadata`
(`_prepare_documents_sync`, document_routes.py:1817-1837, sets only file_id/user_id/
digest/document_origin_type/tenant_id/filename/link/subscription_id/ingest_id). So a
`partial`/`unverified` source's CHUNKS do not carry an index-state field, and a
retrieval-only consumer sees the incomplete state ONLY if it read the /embed receipt.

Therefore the P06-4 outcome "incomplete states survive retrieval" splits into two
provable, contract-accurate parts, and one flagged gap:
  1. On the /embed receipt, partial/unverified surface AS SUCH and are never promoted to
     indexed -- ALREADY PROVEN by tests/utils/test_parse_is_not_index.py (pinned here,
     not re-proven).
  2. The RETRIEVAL path must not FABRICATE a completeness claim: a chunk from a
     partial/unverified write must not come back from /query carrying any
     status/index/complete/indexed field the store never held. THIS FILE proves that.
  3. GAP (reported, not silently satisfied): if the outcome requires the incomplete
     state to reach a retrieval-only consumer, that is a PRODUCER change (attach an
     index-state / partial flag to cmetadata) and is OUT OF SCOPE for P06-4 -- it is
     carded, not manufactured here.

The honest per-chunk provenance that DOES survive retrieval (text_source, ocr_confidence,
ingest_id, locators) is proven by tests/utils/test_query_response_model.py
(text_source="ocr" and an arbitrary future key survive the response model) and
tests/utils/test_mixed_page_extraction.py (ocr chunks carry text_source="ocr"). This
file adds only the ingest_id-survives-and-no-completeness-fabrication assertions so it
does not duplicate those.

Hermetic: FakeStore only, no pgvector needed (runs in the no-DB suite too).
"""
import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes import document_routes
from tests.utils.test_parse_is_not_index import ShortStore, BlindStore, TEXT
from tests.utils.test_replace_not_accumulate import FakeStore

_SECRET = "testsecret"
# P06-2 collection namespace (identity/permission fields it proved survive into rows).
FID = "kn-partial-src"
ENTITY = "ent-knowledge-1"
TENANT = "tenant-vivaldi"

# Any field that would (wrongly) assert completeness/index-state at the chunk level.
_COMPLETENESS_KEYS = (
    "status", "index", "index_status", "indexed", "complete", "completeness",
    "chunks_confirmed", "chunks_prepared", "extraction_status",
)


def _hdr():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "uploader-user", "tid": TENANT, "ent": [ENTITY], "act": ["read", "write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _client(monkeypatch, store):
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)

    # /query reads back exactly what the FakeStore holds for this file.
    async def _retrieve_from_store(*_a, **_k):
        return [(Document(page_content=r.document, metadata=dict(r.metadata)), 0.1)
                for r in store.rows if r.custom_id == FID]
    monkeypatch.setattr(document_routes, "_retrieve_documents", _retrieve_from_store)
    monkeypatch.setattr(document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2])
    return TestClient(app)


def _embed(client, text=TEXT):
    return client.post("/embed", data={"file_id": FID, "entity_id": ENTITY}, headers=_hdr(),
                       files={"file": ("doc.txt", io.BytesIO(text.encode()), "text/plain")})


def _query(client):
    return client.post("/query", json={"query": "q", "file_id": FID, "k": 10,
                                       "entity_id": ENTITY}, headers=_hdr())


def _stored(store):
    return [r for r in store.rows if r.custom_id == FID]


def test_a_partial_write_is_partial_on_the_receipt_and_uninflated_at_retrieval(monkeypatch):
    """A short write reports index.status=partial on the /embed receipt (pin), and the
    SAME chunks retrieved via /query carry NO completeness/index field -- the query path
    neither promotes them to indexed nor fabricates a status the store never held."""
    store = ShortStore()
    client = _client(monkeypatch, store)

    emb = _embed(client)
    assert emb.status_code == 200, emb.text
    idx = emb.json()["index"]
    assert idx["status"] == "partial", ("precondition: the write really landed short", idx)
    assert idx["chunks_confirmed"] < idx["chunks_prepared"], idx

    q = _query(client)
    assert q.status_code == 200, q.text
    hits = q.json()
    assert hits, "precondition: the partial write still stored retrievable chunks"
    for hit in hits:
        meta = hit[0]["metadata"]
        leaked = [k for k in _COMPLETENESS_KEYS if k in meta]
        assert not leaked, (
            "the query path put a completeness/index field on a chunk from a PARTIAL "
            "write: %s -- index-state is a receipt property, retrieval must not assert it"
            % leaked
        )
        # the honest per-chunk provenance the write DID stamp is still there
        assert meta.get("ingest_id"), ("ingest_id provenance lost at retrieval", meta)


def test_an_unverified_write_is_never_promoted_to_indexed_or_complete_at_retrieval(monkeypatch):
    """A store that cannot be read back reports index.status=unverified on the receipt.
    Its chunks retrieved via /query must not come back asserting indexed/complete either
    -- unverified must never be silently upgraded, at write time OR at retrieval."""
    store = BlindStore()
    client = _client(monkeypatch, store)

    emb = _embed(client)
    assert emb.status_code == 200, emb.text
    idx = emb.json()["index"]
    assert idx["status"] == "unverified" and idx["chunks_confirmed"] is None, idx
    assert _stored(store), "precondition: rows WERE stored; only the read-back failed"

    q = _query(client)
    assert q.status_code == 200, q.text
    assert q.json(), "precondition: the unverified write is still retrievable"
    for hit in q.json():
        meta = hit[0]["metadata"]
        for token in ("indexed", "complete"):
            assert token not in {str(v).lower() for v in meta.values()}, (
                "a chunk from an UNVERIFIED write came back asserting %r: %s"
                % (token, meta)
            )
        assert not [k for k in _COMPLETENESS_KEYS if k in meta], meta


def test_the_chunk_never_carries_an_index_status_field_by_construction(monkeypatch):
    """Direct at the producer: _prepare_documents_sync stamps no index-state key, so no
    completeness field can reach a chunk regardless of the store's later read-back. This
    pins the measured contract fact that item 4's 'on cmetadata at retrieval time' rests
    on (and is why the incomplete state lives on the /embed receipt instead)."""
    docs = document_routes._prepare_documents_sync(
        [Document(page_content="a synthetic paragraph of words " * 5, metadata={})],
        FID, ENTITY, False, ingest_id="ing-1",
    )
    assert docs, "precondition: the source split into at least one chunk"
    for d in docs:
        leaked = [k for k in _COMPLETENESS_KEYS if k in d.metadata]
        assert not leaked, ("prepare stamped an index/completeness field on a chunk", leaked)
        assert d.metadata["ingest_id"] == "ing-1"

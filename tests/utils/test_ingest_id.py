"""`ingest_id`: which write a chunk came from (FILES-01 F03, Core C05).

WHY. Core's A08 rule forbids a silent latest-version substitute, and before this a citation
taken before a replace resolved to the NEW text with nothing saying it had changed: rag_api
had no version concept at all. Core decided the contract (Q1 in
.coord/CORE-TO-FILES-CONTRACT-ANSWERS-20260921.md) and these tests pin it exactly:

  * a new UUID per SUCCESSFUL write, stamped on every chunk of that write;
  * returned in `/query` and `/query/{entity_id}` result metadata;
  * a failed or rolled-back replace must not change the ingest_id a reader sees;
  * no other field changes.

The table model is test_replace_not_accumulate's FakeStore (it can hold several versions
and roll them back by row identity), so what is asserted is what the store would hold.
"""

import asyncio
import datetime
import io
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes import document_routes
from tests.utils.test_replace_not_accumulate import FakeRow, FakeStore, V1, V2

_SECRET = "testsecret"
FID = "f03-ingest"


def _hdr():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"], "act": ["read", "write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture()
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", fake)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    return fake


@pytest.fixture()
def client(store, monkeypatch):
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    # /query reads what the table model holds, so the id asserted is the id stored.
    async def retrieve_from_store(*_args, **_kwargs):
        return [(Document(page_content=r.document, metadata=dict(r.metadata)), 0.1)
                for r in store.rows]

    monkeypatch.setattr(document_routes, "_retrieve_documents", retrieve_from_store)
    monkeypatch.setattr(document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2])
    return TestClient(app)


def _embed(client, text, name, replace=None):
    data = {"file_id": FID, "entity_id": "userA"}
    if replace is not None:
        data["replace"] = str(replace).lower()
    return client.post("/embed", data=data, headers=_hdr(),
                       files={"file": (name, io.BytesIO(text.encode()), "text/plain")})


def _ids(store):
    return {r.metadata.get("ingest_id") for r in store.rows if r.custom_id == FID}


def test_every_chunk_of_one_write_carries_the_same_new_uuid(client, store):
    assert _embed(client, V1, "v1.txt").status_code == 200
    rows = [r for r in store.rows if r.custom_id == FID]
    assert len(rows) > 1, "precondition: the fixture splits into several chunks"
    ids = _ids(store)
    assert len(ids) == 1, ids
    (only,) = ids
    assert uuid.UUID(only).version == 4


def test_each_write_gets_its_own_id(client, store):
    assert _embed(client, V1, "v1.txt").status_code == 200
    first = _ids(store)
    assert _embed(client, V2, "v2.txt").status_code == 200  # additive default: both kept
    assert len(_ids(store)) == 2 and first < _ids(store)


def test_a_successful_replace_leaves_only_the_new_id(client, store):
    assert _embed(client, V1, "v1.txt").status_code == 200
    old = _ids(store)
    assert _embed(client, V2, "v2.txt", replace=True).status_code == 200
    now = _ids(store)
    assert len(now) == 1 and now.isdisjoint(old), (old, now)


def test_query_returns_the_stored_ingest_id(client, store):
    """Core reads it from /query and /query/{entity_id}; both must carry it."""
    assert _embed(client, V1, "v1.txt").status_code == 200
    (stored,) = _ids(store)
    # Without this the loop below compares None == None and passes with no stamp at all
    # (measured: the no-stamp mutation left this test green until it was added).
    assert stored, "precondition: the write stamped an ingest_id"
    for path, body in (("/query", {"query": "q", "file_id": FID, "k": 1, "entity_id": "userA"}),
                       ("/query/userA", {"query": "q", "k": 1})):
        r = client.post(path, json=body, headers=_hdr())
        assert r.status_code == 200, (path, r.text)
        hits = r.json()
        assert hits and all(h[0]["metadata"].get("ingest_id") == stored for h in hits), (path, hits)


def test_a_document_cannot_assert_its_own_ingest_id():
    """A service field: loader output must not be able to claim a version."""
    docs = document_routes._prepare_documents_sync(
        [Document(page_content="text", metadata={"ingest_id": "forged-by-the-file"})],
        FID, "userA", False, ingest_id="the-real-one",
    )
    assert {d.metadata["ingest_id"] for d in docs} == {"the-real-one"}


def test_no_other_metadata_key_changes():
    """Core: 'No other field changes.' The only difference the stamp makes is its own key."""
    doc = Document(page_content="text", metadata={"page": 3, "source": "x"})
    without = document_routes._prepare_documents_sync([doc], FID, "userA", False)
    with_id = document_routes._prepare_documents_sync([doc], FID, "userA", False, ingest_id="i")
    extra = set(with_id[0].metadata) - set(without[0].metadata)
    assert extra == {"ingest_id"}
    assert {k: v for k, v in with_id[0].metadata.items() if k != "ingest_id"} == without[0].metadata


class _FailsOnSecondBatch(FakeStore):
    async def aadd_documents(self, docs, ids=None, executor=None):
        self.batches = getattr(self, "batches", 0) + 1
        if self.batches == 2:
            raise RuntimeError("store unavailable mid-write")
        return await super().aadd_documents(docs, ids=ids, executor=executor)


def _seed_prior(store):
    store.rows.append(FakeRow(FID, V1, {"user_id": "userA", "tenant_id": "tenantA",
                                        "ingest_id": "prior-write"}))


def test_a_failed_replace_leaves_the_reader_on_the_prior_id(monkeypatch):
    store = _FailsOnSecondBatch()
    _seed_prior(store)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    with pytest.raises(Exception), ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V2 * 3, metadata={})], FID, "userA",
            executor=pool, tenant_id="tenantA", replace=True,
        ))
    # >= 2: the async pipeline records a failed batch and carries on (pre-existing), then
    # rolls back at the end; what matters is the state a reader is left with.
    assert store.batches >= 2, "precondition: the write failed part-way"
    assert _ids(store) == {"prior-write"}


def test_an_abandoned_replace_leaves_the_reader_on_the_prior_id(monkeypatch):
    store = FakeStore()
    _seed_prior(store)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    answers = iter([True])

    async def leaves_after_insert():
        return next(answers, False)

    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V2, metadata={})], FID, "userA",
            executor=pool, tenant_id="tenantA", replace=True, still_wanted=leaves_after_insert,
        ))
    assert err.value.status_code == document_routes.CALLER_GONE_STATUS
    assert _ids(store) == {"prior-write"}

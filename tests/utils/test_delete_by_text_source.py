"""Delete only the rows one producer wrote, so an escalation can SUPERSEDE (FILES-01).

WHY THIS EXISTS
---------------
When Core escalates a poorly-read scan to the approved AWS route, the better text must
replace the local OCR text rather than sit beside it: two texts for one page make a
citation ambiguous and double-count the same content in retrieval.

The safe ordering is **embed the better text first, then delete what it superseded** --
because the other ordering, delete then embed, leaves the document with ZERO rows if the
second step fails. That ordering was not expressible before this change: the delete path
could filter by file, user, origin type and subscription, but not by which producer wrote
the row, so "delete only the OCR rows for this file" had to become "delete the file".

THE HAZARD THIS FILE IS REALLY GUARDING
---------------------------------------
Every clause in `_delete_multiple` NARROWS. So a filter that arrives and matches nothing
deletes nothing -- harmless. But a filter that is silently DROPPED anywhere between the
request body and the SQL deletes **the whole file**, including the text the caller
explicitly asked to keep. That is unrecoverable data loss from a parameter that looks
present at the top of the call.

The tests HERE cover the seams a unit test can actually reach: that the parameter is
recorded as it crosses into the SQL layer, that an unrecognised value is refused before
the store is touched at all, and that an unfiltered delete is byte-for-byte the call it
always was.

They cannot, on their own, prove the filter is not dropped between the route and the
table -- only the table can answer that. That proof is a separate connected run against
real pgvector, which asserts on which rows SURVIVE:

    .coord/restart-2026-09-16/FILES-01/e2e/e2e_delete_by_text_source.py

It covers supersede (OCR rows go, native rows stay), no-widening (an unknown filter
deletes nothing), the unchanged unfiltered path, and the ordering property Core asked
for: embed the better text first, then remove what it superseded, so the document is
never left with zero rows. All pass.
"""

import datetime
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

_SECRET = "test_key"


def _hdr(action, entity="userA"):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "testuser",
        "tid": "tenantA",
        "ent": [entity],
        "act": [action],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    import jwt

    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


# ---------------------------------------------------------------------------
# Unit level: the SQL clause exists and narrows
# ---------------------------------------------------------------------------


def test_the_filter_reaches_the_sql_layer_with_its_value():
    """The parameter must arrive at `_delete_multiple`, not merely be accepted by the
    route. Recorded at the boundary, because that is where it would be dropped."""
    from unittest.mock import patch

    from app.services.vector_store.async_pg_vector import AsyncPgVector

    seen = {}

    def capture(self, ids=None, collection_only=False, **kwargs):
        seen.update(kwargs)
        seen["ids"] = ids

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    with patch.object(AsyncPgVector, "_delete_multiple", capture):
        store = AsyncPgVector.__new__(AsyncPgVector)
        # `asyncio.run`, not `get_event_loop().run_until_complete`: the latter has no
        # loop to find once other tests in the session have closed theirs, and passed
        # only when this file ran alone.
        with ThreadPoolExecutor(1) as pool:
            asyncio.run(
                AsyncPgVector.delete(
                    store, ids=["f1"], user_id="userA", text_source="ocr", executor=pool
                )
            )

    assert seen.get("text_source") == "ocr", (
        f"text_source never reached the SQL layer; got {seen!r}. A dropped narrowing "
        "filter deletes the WHOLE file."
    )
    assert seen.get("ids") == ["f1"]


# ---------------------------------------------------------------------------
# Route level: validation happens before anything is deleted
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    os.environ["JWT_SECRET"] = _SECRET
    from concurrent.futures import ThreadPoolExecutor

    from main import app

    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


def test_an_unknown_text_source_is_refused_and_deletes_nothing(client, monkeypatch):
    """A typo must be a REFUSAL, never a wider delete.

    This is the whole safety argument in one test: because every filter narrows, an
    unrecognised value that reached the SQL would simply match nothing -- but a value
    mishandled on the way could widen the delete to the entire file. Refusing up front
    means neither can happen, and the caller is told rather than left guessing.
    """
    from app.routes import document_routes

    called = {"n": 0}

    async def must_not_run(*args, **kwargs):
        called["n"] += 1

    monkeypatch.setattr(document_routes.vector_store, "delete", must_not_run, raising=False)

    response = client.request(
        "DELETE",
        "/documents",
        json={"entity_id": "userA", "file_ids": ["f1"], "text_source": "OCR"},
        headers=_hdr("delete"),
    )

    assert response.status_code == 422, response.text
    assert "Unknown text_source" in response.text
    assert "Nothing was deleted" in response.text
    assert called["n"] == 0, "the store was called despite an invalid filter"


def test_a_valid_text_source_is_accepted(client, monkeypatch):
    """The closed set must actually contain the values the loader writes -- otherwise
    the guard above would refuse every legitimate call."""
    from app.routes import document_routes

    assert document_routes._DELETABLE_TEXT_SOURCES == {"native", "ocr"}, (
        "these are the exact values SafePyPDFLoader writes into text_source; they are a "
        "contract surface Core reads at retrieval time"
    )


def test_the_response_does_not_claim_the_file_was_deleted(client, monkeypatch):
    """A partial delete reported as 'deleted successfully' is the same dishonest
    success this lane has spent its whole queue removing."""
    from app.routes import document_routes

    async def fake_ids(*args, **kwargs):
        return ["f1"]

    async def fake_delete(*args, **kwargs):
        return None

    async def fake_summaries(*args, **kwargs):
        return None

    monkeypatch.setattr(document_routes.vector_store, "get_filtered_ids", fake_ids, raising=False)
    monkeypatch.setattr(document_routes.vector_store, "delete", fake_delete, raising=False)
    monkeypatch.setattr(document_routes, "delete_summaries_by_file_ids", fake_summaries)

    response = client.request(
        "DELETE",
        "/documents",
        json={"entity_id": "userA", "file_ids": ["f1"], "text_source": "ocr"},
        headers=_hdr("delete"),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "deleted successfully" not in body["message"], body
    assert "remain" in body["message"], body
    assert body["text_source"] == "ocr"


def test_omitting_the_filter_leaves_the_existing_contract_exactly_as_it_was(
    client, monkeypatch
):
    """Backward compatibility: an ordinary delete must be untouched, in the call it
    makes and in the message it returns."""
    from app.routes import document_routes

    seen = {}

    async def fake_ids(*args, **kwargs):
        return ["f1"]

    async def fake_delete(*args, **kwargs):
        seen.update(kwargs)

    async def fake_summaries(*args, **kwargs):
        return None

    monkeypatch.setattr(document_routes.vector_store, "get_filtered_ids", fake_ids, raising=False)
    monkeypatch.setattr(document_routes.vector_store, "delete", fake_delete, raising=False)
    monkeypatch.setattr(document_routes, "delete_summaries_by_file_ids", fake_summaries)

    response = client.request(
        "DELETE",
        "/documents",
        json={"entity_id": "userA", "file_ids": ["f1"]},
        headers=_hdr("delete"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Documents for 1 file deleted successfully"
    assert seen.get("text_source") is None

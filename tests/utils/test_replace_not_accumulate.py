"""A second upload under the same file_id ADDED rows; both versions stayed retrievable.

WHY THIS FILE EXISTS. Measured 2026-09-20 against the real route path and a real pgvector
store: uploading `policy-v1.txt` under file_id `f3-replace` and then `policy-v2.txt` under
THE SAME file_id left two rows --

    filename=policy-v1.txt  'Travel policy. FS-F3-SUPERSEDED-11871 Limit 500 EUR.'
    filename=policy-v2.txt  'Travel policy. FS-F3-CURRENT-45022  Limit 900 EUR.'

-- and a `/query` returned BOTH. Two different answers to "what is the travel limit",
cited as the same file, with no field on either row saying which is current, and a citation
able to name a filename that no longer exists.

The additive behaviour is deliberate: it is what lets the OCR->native escalation swap text
without the document passing through zero rows. What was missing is replacement WITHIN one
producer. The existing scoped delete keys on `text_source`, a PRODUCER axis, and two .txt
uploads both carry `text_source = None`, so nothing could separate two versions.

`replace=true` captures the file's current ROW PRIMARY KEYS before the insert, inserts,
then deletes exactly those captured rows.

THE CASE THAT DECIDES WHETHER THE IMPLEMENTATION IS RIGHT, and the reason this file models
the table rather than mocking the code: Core pinned it when agreeing the contract. `replace`
must delete the rows captured BEFORE the call -- not "everything except what I just wrote".
When an edited document produces a chunk BYTE-IDENTICAL to one of its own previous chunks
(the ordinary case: one paragraph changed, the rest unchanged), an "except what I wrote"
rule either deletes the row it just created or spares a superseded one.
`test_an_unchanged_chunk_is_replaced_by_its_new_row_not_spared` uses exactly that input.
Two entirely different documents cannot see the difference.

The store double is a small model of `langchain_pg_embedding`: rows with their own primary
key, a shared `custom_id`, content and metadata. It can hold two byte-identical rows under
different keys, which is the whole question.
"""

import datetime
import io
import os
import uuid as uuid_mod
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector

_SECRET = "testsecret"

#: CHUNK_SIZE is 1500, so a one-sentence document is ONE chunk and two versions of it
#: share no chunk at all. These are long enough to split, with an identical leading
#: section and a differing tail, so the replacement really does produce a chunk
#: byte-identical to one of the rows it is superseding. An earlier version of this file
#: used one-line fixtures; the M3 mutation showed the test passed against the WRONG
#: implementation because the condition it describes could not occur.
#: `test_the_fixture_really_produces_an_unchanged_chunk` now asserts that precondition.
_SECTION_ONE = ("Section one of the travel policy is unchanged between versions. " * 30)
V1 = _SECTION_ONE + "\n\nThe reimbursement limit is 500 EUR."
V2 = _SECTION_ONE + "\n\nThe reimbursement limit is 900 EUR."


def _hdr(uid="testuser", tid="tenantA", act=("write",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ["userA"],
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


class FakeRow:
    """One row of langchain_pg_embedding: its OWN key, plus the shared custom_id."""

    def __init__(self, custom_id, document, metadata):
        self.uuid = str(uuid_mod.uuid4())
        self.custom_id = custom_id
        self.document = document
        self.metadata = dict(metadata or {})


class FakeStore(AsyncPgVector):
    """A model of the table, not a mock of the code under test.

    Subclasses the real class so `isinstance(vector_store, AsyncPgVector)` -- which gates
    every async branch in the route -- takes exactly the branch production takes.
    `super().__init__` is deliberately not called: there is no engine and no connection,
    and every method the route reaches is overridden below.

    It records the ORDER of the operations it receives, because ordering is half the
    contract: capture before insert, insert before delete. A double that only answered
    calls would let a wrong order pass.
    """

    def __init__(self):  # noqa: D107 - see class docstring
        # PGVector.__del__ reads `self._bind`; without it, garbage-collecting this double
        # raises an ignored AttributeError that pytest then reports as an unraisable
        # exception against WHICHEVER test happened to trigger the collection. That is a
        # warning attributed to innocent code, so it is prevented here rather than left
        # for someone to chase.
        self._bind = None
        self.rows = []
        self.calls = []

    async def get_row_uuids(self, file_id, user_id=None, tenant_id=None, executor=None):
        self.calls.append("capture")
        return [
            r.uuid
            for r in self.rows
            if r.custom_id == file_id
            and (user_id is None or r.metadata.get("user_id") == user_id)
            and (tenant_id is None or r.metadata.get("tenant_id") == tenant_id)
        ]

    async def delete_rows_by_uuid(self, row_uuids, executor=None):
        self.calls.append("delete")
        self.deleted_arg = list(row_uuids)
        if not row_uuids:
            return 0
        wanted = set(row_uuids)
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.uuid not in wanted]
        return before - len(self.rows)

    async def aadd_documents(self, docs, ids=None, executor=None):
        self.calls.append("insert")
        for d in docs:
            self.rows.append(FakeRow(ids[0] if ids else None, d.page_content, d.metadata))
        return ids

    async def delete(self, *a, **k):
        return None

    def documents_for(self, file_id):
        return [r.document for r in self.rows if r.custom_id == file_id]

    def uuids_for(self, file_id):
        return {r.uuid for r in self.rows if r.custom_id == file_id}

    def filenames_for(self, file_id):
        return sorted({r.metadata.get("filename") for r in self.rows if r.custom_id == file_id})


@pytest.fixture()
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", fake)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    return fake


@pytest.fixture()
def client(store):
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


FID = "f3-replace"


def _embed(client, text, filename, replace=None, file_id=FID):
    data = {"file_id": file_id, "entity_id": "userA"}
    if replace is not None:
        data["replace"] = str(replace).lower()
    return client.post(
        "/embed",
        data=data,
        files={"file": (filename, io.BytesIO(text.encode("utf-8")), "text/plain")},
        headers=_hdr(),
    )


# ---------------------------------------------------------------------------
# The regression this file reproduces
# ---------------------------------------------------------------------------


def test_without_replace_a_second_upload_still_accumulates(client, store):
    """The measured behaviour, pinned so the default cannot change silently.

    This is NOT a defect being preserved: the additive default is what lets the
    OCR->native escalation insert better text before removing what it superseded. It is
    pinned because `replace` must be the caller's explicit choice, never a new default
    that silently starts deleting rows for every existing caller."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    assert _embed(client, V2, "policy-v2.txt").status_code == 200
    docs = store.documents_for(FID)
    assert any("500 EUR" in d for d in docs) and any("900 EUR" in d for d in docs)
    assert store.filenames_for(FID) == ["policy-v1.txt", "policy-v2.txt"]


def test_replace_leaves_only_the_current_version(client, store):
    """THE OUTCOME. After a replacing upload, the superseded text is gone and the
    filename a citation would show is the one that exists."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    r = _embed(client, V2, "policy-v2.txt", replace=True)
    assert r.status_code == 200, r.text

    docs = store.documents_for(FID)
    assert all("500 EUR" not in d for d in docs), docs
    assert any("900 EUR" in d for d in docs), docs
    assert store.filenames_for(FID) == ["policy-v2.txt"]


# ---------------------------------------------------------------------------
# Core's pinned case -- the one that separates two implementations
# ---------------------------------------------------------------------------


def test_the_fixture_really_produces_an_unchanged_chunk(client, store):
    """The precondition of the test below, asserted rather than assumed.

    This exists because the assumption was wrong once. With one-line fixtures each
    version was a single chunk, the two chunks differed, and the byte-identical case the
    next test describes simply never occurred -- so it passed against the very
    implementation it was written to reject. A test whose input cannot exhibit the
    condition is not a weak test, it is a green that could not have failed."""
    assert _embed(client, V1, "policy-v1.txt", file_id="pre-v1").status_code == 200
    v1_chunks = set(store.documents_for("pre-v1"))
    assert len(v1_chunks) > 1, "fixture does not split; CHUNK_SIZE may have changed"

    # A SEPARATE file_id, because the two versions must be compared as two sets. Reading
    # them from one file_id and subtracting removes exactly the shared chunk being looked
    # for, and the assertion then passes over an empty intersection -- which is how this
    # test first reported "no chunk is identical" for fixtures that share 1500 characters.
    assert _embed(client, V2, "policy-v2.txt", file_id="pre-v2").status_code == 200
    v2_chunks = set(store.documents_for("pre-v2"))
    shared = v1_chunks & v2_chunks
    assert shared, (
        "no chunk is byte-identical across the two versions, so the case the next test "
        "describes cannot occur: v1=%d chunks, v2=%d chunks" % (len(v1_chunks), len(v2_chunks))
    )
    assert v1_chunks != v2_chunks, "the two versions are identical; the tail must differ"


def test_an_unchanged_chunk_is_replaced_by_its_new_row_not_spared(client, store):
    """A chunk byte-identical across versions must still be a NEW row, and the OLD row
    with the same text must still go.

    This is the case Core pinned when agreeing the contract, and it is the only one that
    tells the two candidate implementations apart. "Delete everything except what I just
    wrote" would spare the superseded row here, because its text matches a row that was
    just written. Capturing row PRIMARY KEYS before the insert cannot be confused this
    way: the old row's key was recorded before the new row existed.

    Asserted on the keys, not on the text, because the text is identical -- which is
    precisely why text cannot be the identity."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    before = store.uuids_for(FID)
    before_texts = set(store.documents_for(FID))
    assert len(before) > 1, "fixture does not split; there is no unchanged chunk to spare"

    assert _embed(client, V2, "policy-v2.txt", replace=True).status_code == 200
    after = store.uuids_for(FID)
    # The condition this test exists for, restated at the moment it matters: at least one
    # chunk of the new version is byte-identical to one the old version had.
    assert before_texts & set(store.documents_for(FID)), "no unchanged chunk survived"

    # Every row that existed before the call is gone, by key.
    assert before & after == set(), "a superseded row survived: %s" % (before & after)
    # And the delete was aimed at exactly the captured keys.
    assert set(store.deleted_arg) == before
    assert after, "the replacement left the document empty"


def test_the_capture_happens_before_the_insert_and_the_delete_after_it(client, store):
    """Ordering is half the contract and it is invisible in the final state.

    Capture BEFORE insert, or the capture includes the rows just written. Delete AFTER
    insert, or the document passes through zero rows and a reader querying mid-swap sees
    nothing -- the exact failure the OCR escalation ordering exists to avoid."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    store.calls.clear()
    assert _embed(client, V2, "policy-v2.txt", replace=True).status_code == 200
    assert store.calls == ["capture", "insert", "delete"], store.calls


def test_the_document_is_never_empty_at_any_moment_of_the_swap(client, store):
    """Stated as its own assertion because it is the property a reader experiences.

    Checked by counting rows at the moment the delete is issued: the new rows must
    already be present, so the row count never reaches zero between the two versions."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    v1_rows = len(store.rows)

    seen = {}
    original = FakeStore.delete_rows_by_uuid

    async def watching_delete(self, row_uuids, executor=None):
        seen["rows_at_delete"] = len(self.rows)
        return await original(self, row_uuids, executor=executor)

    FakeStore.delete_rows_by_uuid = watching_delete
    try:
        assert _embed(client, V2, "policy-v2.txt", replace=True).status_code == 200
    finally:
        FakeStore.delete_rows_by_uuid = original

    # Both versions present when the delete runs -- strictly more rows than either
    # version alone, so the count can never have passed through zero.
    assert seen["rows_at_delete"] > v1_rows, seen
    assert 0 < len(store.rows) < seen["rows_at_delete"]


# ---------------------------------------------------------------------------
# Failure and honesty
# ---------------------------------------------------------------------------


def test_a_failed_insert_never_removes_the_only_copy(client, store):
    """If the store cannot accept the new version, the old one must survive intact.

    The delete is reached only after the insert returns; an insert that raises must
    propagate before anything is removed. Getting this backwards would turn a transient
    store outage into data loss."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    before = store.uuids_for(FID)

    async def failing_insert(docs, ids=None, executor=None):
        store.calls.append("insert-failed")
        raise RuntimeError("vector store unavailable")

    store.aadd_documents = failing_insert
    r = _embed(client, V2, "policy-v2.txt", replace=True)

    assert r.status_code >= 400, r.text
    assert store.uuids_for(FID) == before, "the superseded rows were removed anyway"
    assert "delete" not in store.calls, store.calls


def test_the_response_reports_what_was_actually_removed(client, store):
    """A replacement that did not remove what it superseded leaves stale content
    retrievable. That has to reach the caller, not be inferred from a 200."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    v1_rows = len(store.uuids_for(FID))
    r = _embed(client, V2, "policy-v2.txt", replace=True)
    rep = r.json()["replacement"]
    assert rep == {
        "superseded_rows": v1_rows, "removed": v1_rows, "status": "complete"
    }, rep


def test_an_incomplete_removal_is_not_reported_as_complete(client, store):
    """The honesty leg. If the delete removes fewer rows than were captured, the caller
    is told `incomplete` -- superseded content is still retrievable and a 200 alone would
    read as "the old version is gone"."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    v1_rows = len(store.uuids_for(FID))

    async def partial_delete(row_uuids, executor=None):
        store.calls.append("delete")
        store.deleted_arg = list(row_uuids)
        return 0  # captured 1, removed 0

    store.delete_rows_by_uuid = partial_delete
    r = _embed(client, V2, "policy-v2.txt", replace=True)
    rep = r.json()["replacement"]
    assert rep["superseded_rows"] == v1_rows and rep["removed"] == 0
    assert rep["status"] == "incomplete", rep


def test_a_first_upload_with_replace_deletes_nothing(client, store):
    """An empty capture means "this file had no rows", which must not become "delete
    everything". The other delete path in this codebase treats a falsy id list as "no id
    filter" and would clear the collection; this one must not."""
    r = _embed(client, V1, "policy-v1.txt", replace=True)
    assert r.status_code == 200, r.text
    assert r.json()["replacement"] == {
        "superseded_rows": 0, "removed": 0, "status": "complete"
    }
    assert store.documents_for(FID), "the first upload stored nothing"


def test_the_replacement_block_is_absent_when_replace_was_not_asked_for(client, store):
    """Every existing response keeps its exact shape. Absent means "nothing was
    superseded", never "we did not check" -- a caller that never asks for replacement
    should not have to learn a new field."""
    r = _embed(client, V1, "policy-v1.txt")
    assert r.status_code == 200
    assert "replacement" not in r.json()


def test_replace_cannot_reach_across_a_tenant_boundary(client, store):
    """The capture is scoped by user and tenant, so a caller passing someone else's
    file_id cannot cause their rows to be deleted. Belt and braces behind the
    entitlement check, because this is the one operation that removes data."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    # Re-label the stored row as another tenant's, leaving the file_id alone.
    for row in store.rows:
        row.metadata["tenant_id"] = "tenantB"
    other = store.uuids_for(FID)

    assert _embed(client, V2, "policy-v2.txt", replace=True).status_code == 200
    assert other <= store.uuids_for(FID), "another tenant's rows were deleted"


def test_embed_upload_honours_replace_too(client, store):
    """A form field the handler does not declare is DROPPED by FastAPI, so a caller
    posting replace=true to /embed-upload would have received a 200 while both versions
    stayed retrievable. Half a feature that answers 200 to the half it does not have is
    worse than not having it at all."""
    def _upload(text, filename, replace=None):
        data = {"file_id": FID, "entity_id": "userA"}
        if replace is not None:
            data["replace"] = str(replace).lower()
        return client.post(
            "/embed-upload",
            data=data,
            files={"uploaded_file": (filename, io.BytesIO(text.encode("utf-8")), "text/plain")},
            headers=_hdr(),
        )

    assert _upload(V1, "policy-v1.txt").status_code == 200
    r = _upload(V2, "policy-v2.txt", replace=True)
    assert r.status_code == 200, r.text
    assert r.json()["replacement"]["status"] == "complete"
    docs = store.documents_for(FID)
    assert all("500 EUR" not in d for d in docs), docs
    assert any("900 EUR" in d for d in docs), docs

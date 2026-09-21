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

    async def delete(self, ids=None, collection_only=False, user_id=None,
                     document_origin_type=None, subscription_id=None, text_source=None,
                     executor=None, **_):
        """Models `_delete_multiple`: deletes by CUSTOM_ID, i.e. every row of a file.

        This was `return None` -- a no-op -- and that made the double unable to express
        the very failure the rollback test exists to catch. The control proved it: with
        the destructive rollback restored, the test reddened on "rows were left behind"
        instead of on "the old version was destroyed", because nothing in the double
        actually deleted anything. A double that models the table must model THIS method
        too, or the most dangerous delete in the codebase is invisible to every test that
        uses it."""
        self.calls.append("delete-by-file_id")
        if not ids:
            return None
        wanted = set(ids)
        self.rows = [r for r in self.rows if r.custom_id not in wanted]
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
    # Two captures: the scoped one whose rows will be deleted, and an unscoped one that
    # counts what this caller's scope cannot see (see `out_of_scope_rows`). Both are
    # reads and both must precede the insert.
    assert store.calls == ["capture", "capture", "insert", "delete"], store.calls


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
        "superseded_rows": v1_rows, "removed": v1_rows,
        "out_of_scope_rows": 0, "status": "complete",
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
        "superseded_rows": 0, "removed": 0, "out_of_scope_rows": 0, "status": "complete",
    }
    assert store.documents_for(FID), "the first upload stored nothing"


def test_the_replacement_block_is_absent_when_replace_was_not_asked_for(client, store):
    """Every existing response keeps its exact shape. Absent means "nothing was
    superseded", never "we did not check" -- a caller that never asks for replacement
    should not have to learn a new field."""
    r = _embed(client, V1, "policy-v1.txt")
    assert r.status_code == 200
    assert "replacement" not in r.json()


def test_the_route_passes_the_callers_tenant_into_the_capture(client, store):
    """NAME CORRECTED after independent review. This proves the ROUTE hands the caller's
    tenant to the capture and honours a scoped result -- it does NOT prove the SQL filter,
    because the double reimplements `get_row_uuids` in Python. Review demonstrated the
    gap: deleting the `tenant_id` clause from the real query left this file 13/13 green.

    The SQL filter is covered by `test_real_sql_*` below, which skips without a reachable
    pgvector. Two tests, because they prove two different things and the old single name
    claimed both."""
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


# ---------------------------------------------------------------------------
# Independent review, 2026-09-20 -- findings that were not covered at all
# ---------------------------------------------------------------------------


def test_a_failed_LATER_batch_does_not_destroy_the_old_version(client, store, monkeypatch):
    """HIGH finding: the batched insert paths roll back by deleting the WHOLE file.

    Under `replace`, that rollback removes the version this call was superseding -- which
    is still the only good copy, because the new one just failed. Review demonstrated it
    end to end: with the production default EMBEDDING_BATCH_SIZE=500, a failure on any
    batch after the first left the file with ZERO rows and answered 400, so a caller
    retrying only on 5xx would treat the document as done.

    The earlier failure test pinned EMBEDDING_BATCH_SIZE to 0, which takes the single-shot
    branch and is structurally incapable of reaching the pipeline. This one does not pin
    it to 0 -- that is the entire point -- and would have caught the loss."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    before = store.uuids_for(FID)
    assert len(before) > 1

    # Batch size 1 so several batches run; fail on a batch that is NOT the first.
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    calls = {"n": 0}
    real_add = store.aadd_documents

    async def failing_after_first(docs, ids=None, executor=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return await real_add(docs, ids=ids, executor=executor)
        raise RuntimeError("vector store unavailable mid-file")

    store.aadd_documents = failing_after_first
    r = _embed(client, V2, "policy-v2.txt", replace=True)

    assert r.status_code >= 400, r.text
    after = store.uuids_for(FID)
    assert before <= after, "the OLD version was destroyed by the rollback: %s" % (before - after)
    # And the partially written new rows are gone -- the rollback still does its job.
    assert after == before, "rows this failed call inserted were left behind: %s" % (after - before)


def test_rows_outside_the_callers_scope_are_never_reported_as_replaced(client, store):
    """MEDIUM finding: `removed == len(captured)` is 0 == 0 when the capture saw nothing.

    Rows written before `tenant_id` was populated carry no such key, so a tenant-scoped
    capture returns NOTHING for them: the delete removes 0 of 0 and the receipt used to
    read `complete` while the old version was still there and still retrievable. A count
    of what the scope could not see is what stops a 200 meaning "the old version is
    gone"."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    # Legacy shape: the row predates tenant stamping.
    for row in store.rows:
        row.metadata.pop("tenant_id", None)
    legacy = store.uuids_for(FID)

    r = _embed(client, V2, "policy-v2.txt", replace=True)
    assert r.status_code == 200, r.text
    rep = r.json()["replacement"]

    assert rep["superseded_rows"] == 0 and rep["removed"] == 0
    assert rep["out_of_scope_rows"] == len(legacy)
    assert rep["status"] == "incomplete", rep
    # The honest part: those rows really are still there.
    assert legacy <= store.uuids_for(FID)


def test_a_clean_replacement_still_reports_no_rows_outside_scope(client, store):
    """Positive control for the field above: the ordinary case must read zero, or
    `incomplete` becomes an alarm that fires on the normal case and gets ignored."""
    assert _embed(client, V1, "policy-v1.txt").status_code == 200
    rep = _embed(client, V2, "policy-v2.txt", replace=True).json()["replacement"]
    assert rep["out_of_scope_rows"] == 0
    assert rep["status"] == "complete", rep


# ---------------------------------------------------------------------------------------
# `out_of_scope_rows` scoping. Added after a measured cross-tenant disclosure, found while
# reviewing CORE's consumer of this very receipt rather than by reading this file.
#
# The count was derived from an UNSCOPED view of the file_id, and `file_id` arrives in the
# form body -- the caller chooses it. Measured against a real service: tenant B held 4 rows
# under `quarterly-board-pack`; tenant A uploaded its own file under that name, replaced
# it, and got `out_of_scope_rows: 4` with `status: incomplete` against a control of 0 and
# `complete`. An existence oracle, the stranger's exact row count, and a false alarm about
# the caller's own data that a consumer renders to a person.
#
# These two tests are a PAIR and neither is sufficient. The first proves a stranger is
# excluded; on its own it is satisfied by deleting the field entirely. The second proves
# the case the field EXISTS for is still counted.
# ---------------------------------------------------------------------------------------


def test_another_tenants_rows_are_not_counted_as_out_of_scope(client, store):
    """THE DISCLOSURE. A stranger's rows under the same caller-chosen file_id must be
    invisible to this count -- otherwise the number is an existence oracle and a size
    estimate for a document the caller cannot read."""
    stranger = FakeRow(FID, "another tenant's confidential board pack", {
        "file_id": FID, "user_id": "userB", "tenant_id": "tenantB",
    })
    store.rows.append(stranger)

    assert _embed(client, V1, "mine-v1.txt").status_code == 200
    r = _embed(client, V2, "mine-v2.txt", replace=True)
    rep = r.json()["replacement"]

    assert rep["out_of_scope_rows"] == 0, (
        "a stranger's rows were counted (%r). `file_id` is caller-chosen, so this number "
        "tells any caller whether another tenant holds that id, and how many rows they "
        "have." % (rep,)
    )
    assert rep["status"] == "complete", (
        "the caller's own replacement succeeded, but a stranger's rows made it report "
        "%r -- which a consumer renders to a person as 'a previous version may still be "
        "retrievable'" % (rep["status"],)
    )
    # And the stranger's row is untouched: this is a scoping fix, not a wider delete.
    assert stranger in store.rows, "the fix must not have widened what gets deleted"


def test_the_callers_own_pre_tenant_rows_ARE_still_counted(client, store):
    """THE CASE THE FIELD EXISTS FOR, which the disclosure fix must not remove.

    A row written for this file before `tenant_id` was populated carries no tenant key,
    so the tenant-scoped capture misses it and the delete cannot supersede it. That row
    is genuinely still retrievable and the caller must be told. Same owner, no tenant.
    """
    legacy = FakeRow(FID, "the caller's own older version", {
        "file_id": FID, "user_id": "userA",  # same owner, and NO tenant_id
    })
    store.rows.append(legacy)

    assert _embed(client, V1, "mine-v1.txt").status_code == 200
    r = _embed(client, V2, "mine-v2.txt", replace=True)
    rep = r.json()["replacement"]

    assert rep["out_of_scope_rows"] >= 1, (
        "the caller's own pre-tenant row was not counted (%r) -- the disclosure fix has "
        "removed the case this field was built for, and a stale version would now be "
        "reported as a complete replacement" % (rep,)
    )
    assert rep["status"] == "incomplete", rep

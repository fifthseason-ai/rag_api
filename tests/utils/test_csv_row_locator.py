"""A CSV row is a citable unit, and a row with no values is not extracted content.

WHY THIS FILE EXISTS. Measured 2026-09-20 against the real route path: a CSV's chunks
already carry `row: N` in the vector store -- langchain's CSVLoader puts it there -- but
the extraction receipt reported `locator_kind: 'none'` and `units_total: 1` for every
CSV, however many rows it had. Two consequences, both measured before this change:

  1. The receipt DENIED a locator the store was already carrying, so a citation could
     name the file and never the row, and `empty_locators` was structurally unable to
     name one.
  2. Worse: a 4-row CSV with two value-less rows produced a BYTE-IDENTICAL receipt to a
     complete one -- same `status: complete`, same counts, same empty `empty_locators`.
     Genuine-zero and partial collapsed into success on fields Core consumers read.

(2) does not follow from (1) and is not fixed by it. CSVLoader renders a value-less row
as the LABELS ALONE ("region: \nrevenue: "), which survives `clean_text(...).strip()`
non-empty, so the row would still have counted as extracted with the locator in place.
The loader now decides emptiness, where the field values are actually known, and yields
that row with EMPTY page_content and its `row` locator intact -- the same shape, and for
the same reason, as an image-only PPTX slide: still citable, never counted as text.

Both legs are asserted here, and each has a positive control that reddens at its own site
(a label-shaped VALUE must not be mistaken for scaffolding; a blank row must not be
dropped and silently renumber the rest).

Fixtures are synthetic and generated at test time; no client content.
"""

import asyncio
import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes.document_routes import _extraction_receipt, load_file_content
from app.services.vector_store.async_pg_vector import AsyncPgVector

_SECRET = "testsecret"

COMPLETE_CSV = "region,revenue\nEMEA,4200000\nAMER,3100000\nAPAC,1800000\n"
#: Rows 1 and 3 (0-indexed, as the loader numbers them) parse fine and carry no values.
#: This is an ordinary export with gaps, NOT a corrupt file -- the distinction the
#: receipt exists to keep.
GAPPY_CSV = "region,revenue\nEMEA,4200000\n,\nAPAC,1800000\n,\n"
#: Every row value-less: the refusal leg.
ALL_BLANK_CSV = "region,revenue\n,\n,\n"
#: A real value that happens to be shaped like the loader's own "label: value"
#: scaffolding. If emptiness were decided by pattern-matching the rendered text instead
#: of by the loader (which knows the field values), this row would be called empty.
LABEL_SHAPED_CSV = "region,note\n,\nEMEA,revenue: 4200000\n"


def _hdr(uid="testuser", tid="tenantA"):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ["userA"],
        "act": ["write"],
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture()
def rec_client(monkeypatch):
    """TestClient with the vector store SIMULATED; every insert is recorded so the
    receipt's counts can be checked against what was ACTUALLY stored."""
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    added = []

    async def recording_aadd(self, docs, ids=None, executor=None):
        added.append(list(docs))
        return ids

    async def dummy_delete(self, ids=None, collection_only=False, user_id=None,
                           document_origin_type=None, subscription_id=None,
                           executor=None, **_):
        return None

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", recording_aadd)
    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)
    client = TestClient(app)
    client.inserted_batches = added  # type: ignore[attr-defined]
    return client


def _embed(client, filename, text):
    return client.post(
        "/embed",
        data={"file_id": "f-csv", "entity_id": "userA"},
        files={"file": (filename, io.BytesIO(text.encode("utf-8")), "text/csv")},
        headers=_hdr(),
    )


def _stored(client):
    return [d for batch in client.inserted_batches for d in batch]


def _load(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    ex = ThreadPoolExecutor(max_workers=2)
    loaded = asyncio.run(load_file_content(name, "text/csv", str(p), ex))
    return loaded[0]


# ---------------------------------------------------------------------------
# Leg 1 -- the row is a unit the receipt can cite
# ---------------------------------------------------------------------------


def test_a_csv_row_is_a_citable_unit_not_the_whole_file(rec_client):
    """Three data rows are three units, not one. Before this change every CSV reported
    units_total=1 regardless of size, so coverage was unreadable."""
    r = _embed(rec_client, "ledger.csv", COMPLETE_CSV)
    assert r.status_code == 200, r.text
    rec = r.json()["extraction"]
    assert rec["locator_kind"] == "row"
    assert (rec["units_total"], rec["units_extracted"]) == (3, 3)
    assert rec["status"] == "complete"


def test_the_locator_the_receipt_names_is_the_one_the_store_carries(rec_client):
    """A citation is only openable if the receipt's locator family matches the stored
    metadata key. Asserted against the ACTUAL recorded inserts, not against intent."""
    assert _embed(rec_client, "ledger.csv", COMPLETE_CSV).status_code == 200
    stored_rows = sorted(d.metadata.get("row") for d in _stored(rec_client))
    assert stored_rows == [0, 1, 2]


# ---------------------------------------------------------------------------
# Leg 2 -- a row with no values is not extracted content
# ---------------------------------------------------------------------------


def test_a_csv_with_value_less_rows_is_partial_not_complete(rec_client):
    """THE REGRESSION THIS FILE REPRODUCES. A gappy file and a complete file used to
    return the same receipt."""
    r = _embed(rec_client, "gappy.csv", GAPPY_CSV)
    assert r.status_code == 200, r.text
    rec = r.json()["extraction"]
    assert rec["status"] == "partial"
    assert (rec["units_total"], rec["units_extracted"], rec["units_empty"]) == (4, 2, 2)
    assert rec["empty_locators"] == [1, 3]
    assert {e["reason"] for e in rec["reasons"]} == {"empty"}


def test_a_gappy_csv_and_a_complete_csv_do_not_produce_the_same_receipt(rec_client):
    """Stated as its own assertion because the defect WAS the equality: two files with
    different content answered identically.

    NOT evidence for leg 2 on its own, and measured to be so: with the row locator in
    place but the blank-row emptying disabled, this test still passes -- `units_total`
    alone differs (3 vs 4). The leg-2 evidence is
    `test_a_csv_with_value_less_rows_is_partial_not_complete`, which reddens under that
    same mutation. Kept because it is the literal statement of the defect, labelled so
    it is never counted twice."""
    full = _embed(rec_client, "ledger.csv", COMPLETE_CSV).json()["extraction"]
    gappy = _embed(rec_client, "gappy.csv", GAPPY_CSV).json()["extraction"]
    keys = ("status", "units_total", "units_extracted", "units_empty", "empty_locators")
    assert {k: full[k] for k in keys} != {k: gappy[k] for k in keys}


def test_a_value_less_row_contributes_no_stored_chunk(rec_client):
    """units_extracted must equal the units that actually contribute chunks -- the
    receipt's own invariant. The loader's "region: \\nrevenue: " scaffolding is not
    content and must not be indexed as if it were."""
    assert _embed(rec_client, "gappy.csv", GAPPY_CSV).status_code == 200
    stored_rows = sorted(d.metadata.get("row") for d in _stored(rec_client))
    assert stored_rows == [0, 2]


def test_a_csv_of_only_value_less_rows_is_refused_with_zero_writes(rec_client):
    """The refusal leg: zero extracted units is the existing 422 path, and nothing is
    written. Before this change this file returned 200 and indexed two chunks of pure
    column labels."""
    r = _embed(rec_client, "blank.csv", ALL_BLANK_CSV)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["extraction"]["status"] == "empty"
    assert _stored(rec_client) == []


def test_the_refusal_does_not_accuse_a_file_that_parsed_perfectly(rec_client):
    """The 422 this change makes newly reachable for CSV must not describe a readable
    file as corrupt, protected or image-only.

    A CSV whose rows are all value-less reaches the empty-extraction guard only BECAUSE
    of this change -- before the row locator existed the file returned 200 and its column
    labels were indexed. A repair that makes a new input class reachable leaves the guard
    at the end of that path untested, so it is tested here rather than assumed."""
    r = _embed(rec_client, "blank.csv", ALL_BLANK_CSV)
    assert r.status_code == 422, r.text
    message = r.json()["detail"]["message"]
    for accusation in ("image-only", "scanned", "corrupt", "password-protected"):
        assert accusation not in message, "accuses a file that parsed: %r" % message
    # It must still say what DID happen, with the count it actually measured.
    assert "2 row(s)" in message and "empty" in message, message


def test_a_file_that_really_is_unreadable_keeps_the_original_message(rec_client):
    """Positive control for the branch above: the generic wording must survive for the
    formats it is true of. A zero-byte .txt has no locator family, takes the else branch,
    and should still be told it may be empty or corrupt."""
    r = rec_client.post(
        "/embed",
        data={"file_id": "f-empty-txt", "entity_id": "userA"},
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=_hdr(),
    )
    assert r.status_code == 422, r.text
    assert "image-only/scanned" in r.json()["detail"]["message"]


# ---------------------------------------------------------------------------
# Positive controls -- each reddens at its own site
# ---------------------------------------------------------------------------


def test_a_value_shaped_like_a_label_is_still_a_value(rec_client):
    """Emptiness is decided by the loader, which knows the field VALUES -- not by
    pattern-matching the rendered "key: value" text. A cell whose content is
    "revenue: 4200000" is real content; if this row were called empty, the rule would be
    guessing at the wrong surface."""
    r = _embed(rec_client, "notes.csv", LABEL_SHAPED_CSV)
    assert r.status_code == 200, r.text
    rec = r.json()["extraction"]
    assert rec["status"] == "partial"
    assert rec["empty_locators"] == [0]
    stored_rows = sorted(d.metadata.get("row") for d in _stored(rec_client))
    assert stored_rows == [1]


def test_a_value_less_row_keeps_its_place_and_never_renumbers_the_file(tmp_path):
    """The empty row stays present with its own locator, exactly like an image-only
    slide. Dropping it would silently shift every later row's citation by one -- the
    citation would then point at the wrong record while looking correct."""
    docs = _load(tmp_path, "gappy.csv", GAPPY_CSV)
    assert [d.metadata.get("row") for d in docs] == [0, 1, 2, 3]
    assert [bool(d.page_content.strip()) for d in docs] == [True, False, True, False]


def test_the_row_family_does_not_capture_other_formats():
    """`row` is added to the locator table; a PDF/PPTX/XLSX document set must keep the
    family it already had, and a locator-less document must stay `none`."""
    assert _extraction_receipt(
        [Document(page_content="a", metadata={"page": 0})]
    )["locator_kind"] == "page"
    assert _extraction_receipt(
        [Document(page_content="a", metadata={"slide_number": 1})]
    )["locator_kind"] == "slide"
    assert _extraction_receipt(
        [Document(page_content="a", metadata={"page_name": "Sales"})]
    )["locator_kind"] == "sheet"
    assert _extraction_receipt(
        [Document(page_content="whole body", metadata={"source": "x"})]
    )["locator_kind"] == "none"

"""`extraction.degraded` -- the degraded-extraction signal, end to end (CARD-P2-01 S1, parts B + E).

THE DEFECT (CARD-DEGRADED-EXTRACTION-SIGNAL, filed 2026-09-23): `locator_kind: "none"` meant two
things -- "this document legitimately has no addressable units" and "the DOCX structured walk hit a
block it could not place and fell back to a flat read". The receipt could not tell them apart, so
"show processing limits honestly" was unsatisfiable: Core cannot render a distinction it is never
told about. Pinned as a limitation at #90 (`test_docx_locator_absence_contract.py`), which this
change converts into asserting the distinction.

THE CARD'S FOUR ACCEPTANCE BULLETS, and where each is proven:
  1. red-first: a fail-safe document is separated from a no-body one --
     `test_docx_locator_absence_contract.py::test_receipt_distinguishes_absent_by_nature_failsafe_and_no_body_SYNTHETIC`
     and `test_walk_failure_is_named_structure_unreadable_SYNTHETIC` below.
  2. the NEGATIVE: a document with no addressable units carries NO signal --
     `test_the_ordinary_no_structure_cases_carry_no_degraded_signal_SYNTHETIC` (TXT, CSV, an
     ordinary DOCX, a header/footer-only DOCX, an empty DOCX) and the route controls.
  3. the signal survives to the STORED CHUNK and the RECEIPT ON THE WIRE, not only loader
     metadata -- `test_degraded_rides_every_prepared_chunk_SYNTHETIC` (the producer that builds
     cmetadata) and `test_degraded_reaches_the_embed_wire_and_the_store_insert_SYNTHETIC` (the
     real /embed route up to `AsyncPgVector.aadd_documents`, the last in-process point before the
     INSERT). LIMIT: no real pgvector round trip here -- the DB write itself is not exercised by
     this suite (conftest neutralises PGVector); `_prepare_documents_sync` preserving loader keys
     is the same mechanism every locator already rides and is pinned elsewhere on real PG.
  4. what Core renders for each case -- not code in this repo; stated in the S1 receipt and the
     P06-5 contract addendum 3 (Core change = S1b).

PART E, the never-reads-complete regression: `test_status_is_never_complete_while_any_shortfall_signal_is_present`
walks EVERY combination of the four shortfall signals the receipt knows (escalation, read bound,
incomplete image coverage, degraded) and asserts `status` is `partial` whenever at least one is
present, and `complete` only when none is -- so a future change that lets any one of them read
`complete` reddens here, including the new one.

Wire names are written as LITERALS on purpose: they are the contract Core reads, so a rename of
the loader constant must redden this file, and a red-first run against a tree without the signal
fails on an ASSERTION, never on an ImportError. SYNTHETIC: every fixture is built at test time.
"""

import datetime
import io
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

os.environ.setdefault("JWT_SECRET", "testsecret")

from app.routes.document_routes import _extraction_receipt, _prepare_documents_sync  # noqa: E402
from app.services.vector_store.async_pg_vector import AsyncPgVector  # noqa: E402
from app.utils.document_loader import SafeDocxLoader, get_loader  # noqa: E402
from app.utils.extraction_budget import ATTEMPTED_KEY, NOT_INCLUDED_KEY, STOPPED_KEY  # noqa: E402
from main import app  # noqa: E402

from tests.utils.test_docx_locator_absence_contract import (  # noqa: E402
    make_header_footer_only_docx_SYNTHETIC,
)
from tests.utils.test_docx_reading_order import DOCX_MIME, _write_docx  # noqa: E402
from tests.utils.test_docx_sdt_content_control import _customxml, _load, _para  # noqa: E402
from tests.utils.test_parser_fitness import make_docx  # noqa: E402

#: The wire contract (P06-5 addendum 3).
DEGRADED_KEY = "extraction_degraded"
BLOCK_UNREPRESENTABLE = "docx_block_unrepresentable"
STRUCTURE_UNREADABLE = "docx_structure_unreadable"


def _failsafe_body(filler_paragraphs: int = 0) -> str:
    """H1 -> w:customXml(LOSTX) -> AFT: the customXml child carries text the walk cannot place, so
    the fail-safe fires. `filler_paragraphs` lengthens the text so it splits into several chunks."""
    filler = "".join(_para("FILLER%04d %s" % (i, "lorem ipsum dolor " * 6)) for i in range(filler_paragraphs))
    return _para("H1") + filler + _customxml(_para("LOSTX")) + _para("AFT")


# ---------------------------------------------------------------------------------------------
# The loader constants ARE the wire literals
# ---------------------------------------------------------------------------------------------


def test_the_loader_constants_are_the_wire_literals():
    """The receipt and Core read these names. A rename is a contract break, not a refactor."""
    from app.utils import document_loader as dl

    assert dl.DEGRADED_KEY == DEGRADED_KEY
    assert dl.DEGRADED_DOCX_BLOCK_UNREPRESENTABLE == BLOCK_UNREPRESENTABLE
    assert dl.DEGRADED_DOCX_STRUCTURE_UNREADABLE == STRUCTURE_UNREADABLE
    assert dl.DEGRADED_REASONS == (BLOCK_UNREPRESENTABLE, STRUCTURE_UNREADABLE)


# ---------------------------------------------------------------------------------------------
# Bullet 1 -- the second reason: the walk itself failing
# ---------------------------------------------------------------------------------------------


def test_walk_failure_is_named_structure_unreadable_SYNTHETIC(tmp_path, monkeypatch):
    """RED-FIRST. The generic `except Exception` arm of `_structured_units` is the OTHER way a
    DOCX degrades to the flat read. Before the signal existed it was indistinguishable from a
    no-body document. Injected: `_body_units` raises -- the walk fails, the flat read succeeds,
    the text is kept, and the receipt names the reason."""
    _path, loader = _load(tmp_path, "walkfail.docx", _para("KEEP_A") + _para("KEEP_B"))

    def _boom(cls, root):
        raise RuntimeError("synthetic structured-walk failure")

    monkeypatch.setattr(SafeDocxLoader, "_body_units", classmethod(_boom))
    docs = loader.load()
    joined = "\n".join(d.page_content for d in docs)
    assert "KEEP_A" in joined and "KEEP_B" in joined, "the flat read must keep the text"

    receipt = _extraction_receipt(docs)
    assert receipt["locator_kind"] == "none"
    assert receipt.get("degraded", {}).get("reason") == STRUCTURE_UNREADABLE, receipt
    assert receipt["status"] == "partial", receipt
    assert all(d.metadata.get(DEGRADED_KEY) == STRUCTURE_UNREADABLE for d in docs)


def test_a_structured_read_after_a_degraded_one_carries_no_stale_reason_SYNTHETIC(tmp_path):
    """The reason is per-read state on the loader. A loader that degraded once must not carry
    that reason into a later successful walk -- a stale alarm is the normal-case alarm again."""
    _path, loader = _load(tmp_path, "twice.docx", _failsafe_body())
    first = loader.load()
    assert all(d.metadata.get(DEGRADED_KEY) == BLOCK_UNREPRESENTABLE for d in first)
    # Rewrite the SAME path as an ordinary document and read it again with the SAME loader.
    _write_docx(_path, _para("PLAIN_A") + _para("PLAIN_B"))
    second = loader.load()
    assert not any(DEGRADED_KEY in d.metadata for d in second), [d.metadata for d in second]
    assert "degraded" not in _extraction_receipt(second)


# ---------------------------------------------------------------------------------------------
# Bullet 2 -- THE NEGATIVE: the normal no-structure cases never carry the signal
# ---------------------------------------------------------------------------------------------


def _txt(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("plain text with no addressable units at all\nsecond line\n", encoding="utf-8")
    return get_loader("notes.txt", "text/plain", str(p))[0].load()


def _csv(tmp_path):
    p = tmp_path / "rows.csv"
    p.write_text("name,amount\nalpha,1\nbeta,2\n", encoding="utf-8")
    return list(get_loader("rows.csv", "text/csv", str(p))[0].lazy_load())


def _docx_ordinary(tmp_path):
    p = tmp_path / "report.docx"
    make_docx(str(p))
    return get_loader("report.docx", DOCX_MIME, str(p))[0].load()


def _docx_header_footer_only(tmp_path):
    p = tmp_path / "hf_only.docx"
    make_header_footer_only_docx_SYNTHETIC(str(p))
    return get_loader("hf_only.docx", DOCX_MIME, str(p))[0].load()


def _docx_empty_body(tmp_path):
    _path, loader = _load(tmp_path, "empty.docx", "")
    return loader.load()


@pytest.mark.parametrize(
    "build",
    [_txt, _csv, _docx_ordinary, _docx_header_footer_only, _docx_empty_body],
    ids=["txt_no_units", "csv_rows", "docx_structured", "docx_header_footer_only", "docx_empty_body"],
)
def test_the_ordinary_no_structure_cases_carry_no_degraded_signal_SYNTHETIC(tmp_path, build):
    """THE BINDING NEGATIVE. Every one of these reads exactly as designed -- including the two
    DOCX shapes that take the SAME flat fallback as the fail-safe (no body block; an empty body).
    None may carry the signal, at the chunk or on the receipt, and none may be downgraded by it.
    An alarm that fires on the normal case teaches every reader to ignore it."""
    docs = build(tmp_path)
    assert docs, "the loader produced nothing, so this negative would pass for the wrong reason"
    stamped = [d.metadata for d in docs if DEGRADED_KEY in (d.metadata or {})]
    assert not stamped, "an ordinary read carries the degraded stamp: %r" % stamped
    receipt = _extraction_receipt(docs)
    assert "degraded" not in receipt, receipt


# ---------------------------------------------------------------------------------------------
# Bullet 3 -- the signal survives to the stored chunk and the wire
# ---------------------------------------------------------------------------------------------


def test_degraded_rides_every_prepared_chunk_SYNTHETIC(tmp_path):
    """`_prepare_documents_sync` builds the cmetadata every stored row carries. The flat
    Document is SPLIT into several chunks here; each must carry the reason, or a citation to the
    second chunk would read as an ordinary no-locator passage."""
    _path, loader = _load(tmp_path, "long_failsafe.docx", _failsafe_body(filler_paragraphs=40))
    docs = loader.load()
    chunks = _prepare_documents_sync(
        docs, file_id="f-deg", user_id="userA", clean_content=False,
        filename="long_failsafe.docx", tenant_id="tenantA", ingest_id="ing-deg-1",
    )
    assert len(chunks) > 1, "the fixture must split into several chunks for this to prove anything"
    missing = [c.metadata for c in chunks if c.metadata.get(DEGRADED_KEY) != BLOCK_UNREPRESENTABLE]
    assert not missing, "chunks lost the degraded stamp on the way to storage: %r" % missing[:2]
    # The service fields still win and the loader stamp does not displace them.
    assert all(c.metadata["file_id"] == "f-deg" and c.metadata["user_id"] == "userA" for c in chunks)


def _hdr(act):
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"], "act": act,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")}


@pytest.fixture()
def recording_store(monkeypatch):
    """The real /embed route up to `AsyncPgVector.aadd_documents` -- the last in-process point
    before the INSERT -- with a recording double there. No pgvector, no provider."""
    added = []

    async def recording_aadd(self, docs, ids=None, executor=None):
        added.append(list(docs))
        return ids

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", recording_aadd)
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return added


def _embed(client, name, data):
    return client.post(
        "/embed",
        data={"file_id": "f-" + name, "entity_id": "userA"},
        files={"file": (name, io.BytesIO(data), DOCX_MIME)},
        headers=_hdr(["write"]),
    )


def test_degraded_reaches_the_embed_wire_and_the_store_insert_SYNTHETIC(tmp_path, recording_store):
    """RED-FIRST at route level. A consumer never sees loader metadata; it sees the /embed body
    and, later, the stored chunk. Both must carry the signal."""
    path = tmp_path / "failsafe.docx"
    _write_docx(str(path), _failsafe_body())
    r = _embed(TestClient(app), "failsafe.docx", path.read_bytes())
    assert r.status_code == 200, r.text
    extraction = r.json()["extraction"]
    assert extraction.get("degraded", {}).get("reason") == BLOCK_UNREPRESENTABLE, extraction
    assert extraction["degraded"]["scope"] == "document"
    assert extraction["degraded"]["lost"] == "unit_locators"
    assert extraction["degraded"]["units_affected"] == extraction["units_total"]
    assert extraction["status"] == "partial", extraction
    inserted = [d.metadata for batch in recording_store for d in batch]
    assert inserted, "nothing reached the store-insert boundary"
    assert all(m.get(DEGRADED_KEY) == BLOCK_UNREPRESENTABLE for m in inserted), inserted[:2]


def test_the_no_body_document_reaches_the_wire_with_no_signal_SYNTHETIC(tmp_path, recording_store):
    """The route-level NEGATIVE, same harness: header/footer-only takes the same flat fallback
    and must arrive clean -- no receipt block, no chunk stamp, status not downgraded."""
    path = tmp_path / "hf_only.docx"
    make_header_footer_only_docx_SYNTHETIC(str(path))
    r = _embed(TestClient(app), "hf_only.docx", path.read_bytes())
    assert r.status_code == 200, r.text
    extraction = r.json()["extraction"]
    assert "degraded" not in extraction, extraction
    assert extraction["status"] == "complete", extraction
    inserted = [d.metadata for batch in recording_store for d in batch]
    assert inserted, "nothing reached the store-insert boundary"
    assert not any(DEGRADED_KEY in m for m in inserted), inserted


def test_the_text_route_returns_a_typed_receipt_including_degraded_SYNTHETIC(tmp_path):
    """RED-FIRST. `/text` returned the string and nothing else, so an empty, partial or degraded
    read was a 200 whose only signal was the text itself. It now carries the SAME receipt the
    embed routes return. /text stores nothing and never refused an empty read; it still does not."""
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    client = TestClient(app)

    def _text(name, data, mime):
        return client.post(
            "/text", data={"file_id": "t-" + name, "entity_id": "userA"},
            files={"file": (name, io.BytesIO(data), mime)}, headers=_hdr(["read"]),
        )

    failsafe = tmp_path / "failsafe.docx"
    _write_docx(str(failsafe), _failsafe_body())
    r = _text("failsafe.docx", failsafe.read_bytes(), DOCX_MIME)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "LOSTX" in body["text"]
    assert body.get("extraction", {}).get("degraded", {}).get("reason") == BLOCK_UNREPRESENTABLE, body
    assert body["extraction"]["status"] == "partial"

    hf_only = tmp_path / "hf_only.docx"
    make_header_footer_only_docx_SYNTHETIC(str(hf_only))
    r2 = _text("hf_only.docx", hf_only.read_bytes(), DOCX_MIME)
    assert r2.status_code == 200, r2.text
    assert "extraction" in r2.json(), r2.json()
    assert "degraded" not in r2.json()["extraction"]

    # EMPTY is typed on /text too, and still a 200 (unchanged status).
    r3 = _text("blank.txt", b"   \n\n  ", "text/plain")
    assert r3.status_code == 200, r3.text
    assert r3.json()["extraction"]["status"] == "empty", r3.json()


# ---------------------------------------------------------------------------------------------
# The roll-up never silently drops a second reason
# ---------------------------------------------------------------------------------------------


def test_a_second_reason_is_listed_never_dropped():
    """No loader produces two reasons in one read today. The roll-up still must not lose one if
    that ever changes: `reason` is the first seen, `reasons` lists all, in document order."""
    docs = [
        Document(page_content="alpha", metadata={DEGRADED_KEY: STRUCTURE_UNREADABLE}),
        Document(page_content="beta", metadata={DEGRADED_KEY: BLOCK_UNREPRESENTABLE}),
        Document(page_content="gamma", metadata={DEGRADED_KEY: STRUCTURE_UNREADABLE}),
    ]
    degraded = _extraction_receipt(docs)["degraded"]
    assert degraded["reason"] == STRUCTURE_UNREADABLE
    assert degraded["reasons"] == [STRUCTURE_UNREADABLE, BLOCK_UNREPRESENTABLE]

    one = _extraction_receipt([Document(page_content="x", metadata={DEGRADED_KEY: STRUCTURE_UNREADABLE})])
    assert "reasons" not in one["degraded"], "a single reason must not grow a redundant list"


# ---------------------------------------------------------------------------------------------
# PART E -- nothing short, stopped, escalated or degraded ever reads `complete`
# ---------------------------------------------------------------------------------------------

#: Each shortfall signal as the LOADER stamps it on an otherwise fully-extracted page, and the
#: receipt key that must then appear. One page (`page: 0`) with real text, so without any signal
#: the receipt is `complete` -- the control below proves it.
_SIGNALS = {
    "escalated": ({"ocr_reason": "ocr_low_confidence", "ocr_attempted": True,
                   "ocr_confidence": 0.41, "text_source": "ocr"}, "escalation"),
    "bounded": ({STOPPED_KEY: "page_limit", ATTEMPTED_KEY: 1, NOT_INCLUDED_KEY: 4}, "extraction_bound"),
    "image_uncovered": ({"image_ocr_coverage": "not_attempted"}, "coverage"),
    "degraded": ({DEGRADED_KEY: BLOCK_UNREPRESENTABLE}, "degraded"),
}


def _page(extra):
    return [Document(page_content="every page of this document yielded real text",
                     metadata={"page": 0, **extra})]


def _shortfall_present(receipt):
    """The receipt's OWN statement that something fell short -- read off the receipt, so the
    invariant is checked against what a consumer sees, not against what the test injected."""
    return bool(
        (receipt.get("escalation") or {}).get("recommended")
        or "extraction_bound" in receipt
        or (receipt.get("coverage") or {}).get("image_ocr") in ("unknown", "not_attempted")
        or "degraded" in receipt
    )


def test_the_control_reads_complete_so_the_invariant_is_not_vacuous():
    receipt = _extraction_receipt(_page({}))
    assert receipt["status"] == "complete", receipt
    assert not _shortfall_present(receipt)


@pytest.mark.parametrize(
    "combo",
    [c for n in range(1, len(_SIGNALS) + 1) for c in itertools.combinations(sorted(_SIGNALS), n)],
    ids=lambda c: "+".join(c),
)
def test_status_is_never_complete_while_any_shortfall_signal_is_present(combo):
    """Every non-empty combination of the four signals. RED-FIRST: on a tree without the
    degraded signal, every combination containing `degraded` ALONE reads `complete` (nothing
    else forces it) and fails here; the other combinations already held and are pinned."""
    extra = {}
    for name in combo:
        extra.update(_SIGNALS[name][0])
    receipt = _extraction_receipt(_page(extra))
    for name in combo:
        key = _SIGNALS[name][1]
        assert key in receipt, "%s: the receipt lost the %r block: %r" % ("+".join(combo), key, receipt)
    assert _shortfall_present(receipt), receipt
    assert receipt["status"] == "partial", (
        "%s: the receipt reads %r while it itself reports a shortfall -- a consumer gating on "
        "`status` would call this document complete: %r" % ("+".join(combo), receipt["status"], receipt)
    )

"""Extraction-status receipt tests for the /embed pipeline (KI-02 WP-G1).

Directive 7.4 item 4 (verbatim): "Empty or partial extraction must have explicit
status, not success." WP-C proved EMPTY thoroughly (422 + zero writes); this
increment adds the missing PARTIAL leg: an additive `extraction` receipt on the
/embed response (and on the 422 body for empty), derived ONLY from the loader
signals that already exist — never success-by-default.

Receipt shape (see app.routes.document_routes._extraction_receipt):
    extraction: {
      status: 'complete' | 'partial' | 'empty',
      locator_kind: 'page' | 'slide' | 'sheet' | 'row' | 'section' | 'block' | 'none',
                    # one per family in _UNIT_LOCATOR_KEYS; the tuple decides
      units_total, units_extracted, units_empty, units_image_only,
      empty_locators: [...],           # locator-bearing units with no text
      reasons: [{locator, reason}],     # 'image_only' | 'empty' per non-extracted unit
    }

    * partial = at least one extracted unit AND at least one empty/image-only unit,
    * empty   = zero extracted units (the existing 422 path; zero vector writes),
    * units_extracted must equal the units that actually contribute stored chunks.

All fixtures are SYNTHETIC and generated at test time (SYN-KNOWLEDGE-01 label);
no client content. The vector store is SIMULATED (AsyncPgVector.aadd_documents is
recorded) so the receipt counts can be checked against the ACTUAL stored chunks.

Honesty rule proven here: PDF has no image-only signal at the loader (a scanned
page and a blank page are both empty page_content), so PDF empty pages report
`empty`, never `image_only`; only PPTX marks image_only.
"""

import datetime
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes.document_routes import _extraction_receipt
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import SafeDocxLoader

# Reuse the synthetic generators / skip marks already proven in WP-C.
from tests.utils.test_parser_fitness import (
    _add_image_only_slide,
    make_all_image_pptx,
    make_corrupt_ooxml,
    make_docx,
    make_multisheet_xlsx,
)

_SECRET = "testsecret"


# ---------------------------------------------------------------------------
# Synthetic fixture generators new to WP-G1 (label SYN-KNOWLEDGE-01)
# ---------------------------------------------------------------------------


def make_partial_pdf(path):
    """5 pages: pages 0,1,2 carry native text; pages 3,4 are blank (no text
    layer) — the image-only/scan case a PDF cannot distinguish from truly blank.
    Proves `partial` (3 extracted + 2 empty) with the two empty locators."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=200, height=200)
    for i in (0, 1, 2):
        page = writer.pages[i]
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        res = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font
        res[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = res
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 20 100 Td (SYN-KNOWLEDGE-01 native page {i} revenue) Tj ET".encode()
        )
        page[NameObject("/Contents")] = stream
    with open(path, "wb") as f:
        writer.write(f)


def make_partial_pptx(path):
    """Slide 1 = title + table + speaker notes (all extractable); slide 2 =
    image-only (a picture, no text). Proves `partial`, with notes/table text
    stored for slide 1 and slide 2 reported image_only."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title-only layout
    slide.shapes.title.text = "SYN-KNOWLEDGE-01 Quarterly Review"
    table = slide.shapes.add_table(
        2, 2, Inches(1), Inches(2), Inches(4), Inches(1)
    ).table
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "EMEA"
    table.cell(1, 1).text = "EUR 4.2M"
    slide.notes_slide.notes_text_frame.text = (
        "SYN-KNOWLEDGE-01 note: EMEA outperformed."
    )
    _add_image_only_slide(prs)  # slide 2: picture only, no text
    prs.save(path)


def make_markdown(path):
    """A minimal markdown document with a heading and a paragraph."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SYN-KNOWLEDGE-01 Report\n\nRevenue grew twelve percent in EMEA.\n")


# ===========================================================================
# Pure-function unit tests for _extraction_receipt (no route, fast, precise)
# ===========================================================================


def test_receipt_pdf_partial_from_loader_signals():
    """A page-locator doc set with 3 text pages + 2 empty pages -> partial, the
    two empty pages listed as `empty` locators. PDF never reports image_only."""
    docs = [
        Document(page_content=f"page {i} text", metadata={"source": "x", "page": i})
        for i in (0, 1, 2)
    ] + [
        Document(page_content="", metadata={"source": "x", "page": i})
        for i in (3, 4)
    ]
    r = _extraction_receipt(docs)
    assert r["status"] == "partial"
    assert r["locator_kind"] == "page"
    assert r["units_total"] == 5
    assert r["units_extracted"] == 3
    assert r["units_empty"] == 2
    assert r["units_image_only"] == 0  # PDF cannot claim image_only honestly
    assert r["empty_locators"] == [3, 4]
    assert r["reasons"] == [
        {"locator": 3, "reason": "empty"},
        {"locator": 4, "reason": "empty"},
    ]


def test_receipt_pptx_partial_marks_image_only():
    """A slide set with one text slide + one image-only slide -> partial, the
    image-only slide counted as image_only (not empty)."""
    docs = [
        Document(
            page_content="Region | Revenue",
            metadata={"source": "x", "slide_number": 1},
        ),
        Document(
            page_content="",
            metadata={"source": "x", "slide_number": 2, "image_only": True},
        ),
    ]
    r = _extraction_receipt(docs)
    assert r["status"] == "partial"
    assert r["locator_kind"] == "slide"
    assert r["units_total"] == 2
    assert r["units_extracted"] == 1
    assert r["units_empty"] == 0
    assert r["units_image_only"] == 1
    assert r["empty_locators"] == [2]
    assert r["reasons"] == [{"locator": 2, "reason": "image_only"}]


def test_receipt_complete_when_every_unit_extracted():
    docs = [
        Document(page_content="a", metadata={"page": 0}),
        Document(page_content="b", metadata={"page": 1}),
    ]
    r = _extraction_receipt(docs)
    assert r["status"] == "complete"
    assert (r["units_total"], r["units_extracted"]) == (2, 2)
    assert r["empty_locators"] == [] and r["reasons"] == []


def test_receipt_none_locator_single_unit_complete():
    """DOCX/MD/TXT expose no per-unit locator -> one `none` unit; a document with
    text is `complete` with locator_kind none (honest: nothing finer to cite)."""
    docs = [Document(page_content="whole document body", metadata={"source": "x"})]
    r = _extraction_receipt(docs)
    assert r["status"] == "complete"
    assert r["locator_kind"] == "none"
    assert (r["units_total"], r["units_extracted"]) == (1, 1)


def test_receipt_empty_when_no_unit_has_text():
    """Zero extracted units -> status empty (this is the 422 path). NUL-only
    content cleans to empty, matching the guard's clean_text predicate exactly."""
    assert _extraction_receipt([])["status"] == "empty"
    r = _extraction_receipt([Document(page_content="\x00\x00\x00", metadata={"page": 0})])
    assert r["status"] == "empty"
    assert r["units_extracted"] == 0


# ===========================================================================
# Route-level tests: receipt on the 200 body / 422 body, and count↔write parity
# ===========================================================================


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
    return {
        "Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"
    }


@pytest.fixture()
def rec_client(monkeypatch):
    """TestClient with the vector store SIMULATED; every insert is recorded so
    receipt counts can be checked against the ACTUAL stored chunks."""
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    added = []

    async def recording_aadd(self, docs, ids=None, executor=None):
        added.append(list(docs))
        return ids

    async def dummy_delete(self, ids=None, collection_only=False, user_id=None,
                           document_origin_type=None, subscription_id=None, executor=None, **_):
        return None

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", recording_aadd)
    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)

    client = TestClient(app)
    client.inserted_batches = added  # type: ignore[attr-defined]
    return client


def _embed(client, filename, content, content_type):
    return client.post(
        "/embed",
        data={"file_id": "f-recv", "entity_id": "userA"},
        files={"file": (filename, io.BytesIO(content), content_type)},
        headers=_hdr(),
    )


def _stored_docs(client):
    return [d for batch in client.inserted_batches for d in batch]


def test_embed_partial_pdf_reports_partial_and_counts_match_writes(rec_client, tmp_path):
    """A 3-text + 2-empty PDF must return 200 with status 'partial', list the two
    empty page locators, and — critically — units_extracted must equal the number
    of pages that actually produced stored chunks (never success-by-default)."""
    path = tmp_path / "partial.pdf"
    make_partial_pdf(str(path))
    r = _embed(rec_client, "partial.pdf", path.read_bytes(), "application/pdf")
    assert r.status_code == 200, r.text

    rec = r.json()["extraction"]
    assert rec["status"] == "partial"
    assert rec["locator_kind"] == "page"
    assert rec["units_total"] == 5
    assert rec["units_extracted"] == 3
    assert rec["units_empty"] == 2
    assert rec["units_image_only"] == 0
    assert rec["empty_locators"] == [3, 4]

    # Count↔write parity: only the 3 native pages produced stored chunks.
    stored_pages = {d.metadata.get("page") for d in _stored_docs(rec_client)}
    assert stored_pages == {0, 1, 2}
    assert rec["units_extracted"] == len(stored_pages)


def test_embed_partial_pptx_stores_notes_table_and_marks_image_only(rec_client, tmp_path):
    """A deck with a rich slide (title+table+notes) + an image-only slide must be
    `partial`; the notes and table text are stored for the extracted slide, and
    only that slide's locator appears in the written chunks."""
    path = tmp_path / "partial.pptx"
    make_partial_pptx(str(path))
    r = _embed(
        rec_client,
        "partial.pptx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert r.status_code == 200, r.text

    rec = r.json()["extraction"]
    assert rec["status"] == "partial"
    assert rec["locator_kind"] == "slide"
    assert rec["units_total"] == 2
    assert rec["units_extracted"] == 1
    assert rec["units_image_only"] == 1
    assert rec["units_empty"] == 0
    assert rec["empty_locators"] == [2]
    assert rec["reasons"] == [{"locator": 2, "reason": "image_only"}]

    stored = _stored_docs(rec_client)
    stored_slides = {d.metadata.get("slide_number") for d in stored}
    assert stored_slides == {1}
    assert rec["units_extracted"] == len(stored_slides)
    joined = " ".join(d.page_content for d in stored)
    assert "Region" in joined and "EUR 4.2M" in joined  # table cells stored
    assert "[Notes]" in joined and "EMEA outperformed" in joined  # notes stored


def test_embed_all_image_pptx_is_empty_receipt_on_422_no_rows(rec_client, tmp_path):
    """An all-image deck stays the existing 422 path with ZERO writes, and the
    422 body now carries the receipt: status 'empty', every slide image_only."""
    path = tmp_path / "allimg.pptx"
    make_all_image_pptx(str(path))
    r = _embed(
        rec_client,
        "allimg.pptx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "No extractable text" in detail["message"]  # original message preserved
    rec = detail["extraction"]
    assert rec["status"] == "empty"
    assert rec["units_extracted"] == 0
    assert rec["units_total"] == 2
    assert rec["units_image_only"] == 2
    assert rec["empty_locators"] == [1, 2]
    assert rec_client.inserted_batches == []  # zero vector writes, unchanged


def test_embed_docx_reports_block_locator_family_units_and_writes(rec_client, tmp_path):
    """FLIPPED for E1 (was ..._reports_complete_locator_none_and_writes): DOCX now
    carries the per-unit block locator family, so `/embed` reports
    locator_kind == the family, one unit per authored block, status 'complete';
    rows are written and the receipt is present on the 200 body (consumer-shape lock).

    make_docx builds a heading + a body paragraph + a 2x2 table + a header + a footer.
    SafeDocxLoader emits one BODY unit per direct body `w:p` and one per table cell
    (row-major), each with a distinct 0-based block_index (2 paragraphs + 4 table
    cells = 6 body blocks), and it emits the header/footer text as unit(s) carrying
    NO block_index (ruling 2026-09-23: header/footer have no body position). In the
    receipt's detection loop the header/footer chunks fold into the single unnamed
    (None) unit, so units_total = 6 distinct block_index values + 1 = 7.

    NO hardcoded literal is asserted: units_total is checked against the DISTINCT
    locator groups the STORED chunks actually form (each block_index, plus the None
    group for the keyless header/footer chunks), so a loader that dropped a block,
    merged two, or re-stamped a header would move this number. locator_kind is read
    from the loader's own RULED constant; the receipt derives its kind from the
    _UNIT_LOCATOR_KEYS registry, so this also proves registry<->loader agreement."""
    path = tmp_path / "report.docx"
    make_docx(str(path))
    r = _embed(
        rec_client,
        "report.docx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "extraction" in body  # receipt present on the success body
    rec = body["extraction"]
    assert rec["status"] == "complete"
    assert rec["locator_kind"] == SafeDocxLoader._DOCX_LOCATOR_KIND

    # units_total equals the number of DISTINCT locator groups across the stored
    # chunks -- each block_index value, PLUS the None group holding the keyless
    # header/footer chunks (dict.get returns None for a missing key, exactly as the
    # receipt's own grouping does). No integer literal is hardcoded.
    key = SafeDocxLoader._DOCX_LOCATOR_KEY
    stored = _stored_docs(rec_client)
    locs = {(d.metadata or {}).get(key) for d in stored}  # includes None for header/footer
    assert rec["units_total"] == len(locs)
    assert rec["units_total"] > 1  # DOCX no longer folds to a single unit

    body_idx = sorted(v for v in locs if v is not None)
    assert body_idx == list(range(len(body_idx)))  # 0-based contiguous body blocks
    assert None in locs  # the unnamed header/footer unit is present

    # ABSOLUTE anchor (reviewer Finding 2): make_docx has 6 body blocks -- 1 heading
    # + 1 body paragraph + 4 table cells (2x2). Pinned as a literal at ONE site, the
    # way the card fixture is anchored to 9 named tokens. This is what the derived
    # `units_total == len(locs)` agreement above cannot catch on its own: a
    # regression collapsing all body indices to a single value moves BOTH sides of
    # that equality together and still passes, but would break this count.
    # (units_total itself is NOT asserted as a literal -- ruling (c): the test must
    # not name that number; it is pinned via the derived agreement + this body count
    # + the None group, i.e. 6 body blocks + 1 unnamed header/footer unit.)
    assert len(body_idx) == 6, f"expected 6 body blocks for make_docx, got {body_idx}"

    assert rec["units_extracted"] == rec["units_total"]  # every unit text-bearing
    assert rec["empty_locators"] == []
    assert len(rec_client.inserted_batches) >= 1  # rows written


def test_embed_markdown_reports_complete(rec_client, tmp_path):
    """A markdown file extracts to text -> complete. (Since PACKET-1 E4 markdown carries a
    per-section `section` locator; this test asserts only the complete-status/units contract,
    not the locator family -- see tests/utils/test_md_heading_locator.py for the locator.)"""
    path = tmp_path / "note.md"
    make_markdown(str(path))
    r = _embed(rec_client, "note.md", path.read_bytes(), "text/markdown")
    assert r.status_code == 200, r.text
    rec = r.json()["extraction"]
    assert rec["status"] == "complete"
    assert rec["units_extracted"] == rec["units_total"] >= 1


def test_embed_zero_byte_txt_is_empty_receipt_422_no_rows(rec_client):
    """A zero-byte file extracts to nothing -> existing 422 path, status 'empty',
    no rows."""
    r = _embed(rec_client, "empty.txt", b"", "text/plain")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["extraction"]["status"] == "empty"
    assert rec_client.inserted_batches == []


def test_embed_truncated_zip_pptx_is_honest_failure_no_rows(rec_client, tmp_path):
    """A truncated/corrupt OOXML fails honestly (4xx) BEFORE the guard — no
    receipt is fabricated for bytes that never parsed into units, and no rows are
    written."""
    path = tmp_path / "corrupt.pptx"
    make_corrupt_ooxml(str(path))
    r = _embed(
        rec_client,
        "corrupt.pptx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert 400 <= r.status_code < 500, r.text
    assert rec_client.inserted_batches == []


def test_embed_complete_pdf_reports_complete(rec_client, tmp_path):
    """Control: a PDF whose every page has text -> status 'complete', no empty
    locators, and every page produces a stored chunk."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    path = tmp_path / "full.pdf"
    writer = PdfWriter()
    for i in range(3):
        writer.add_blank_page(width=200, height=200)
        page = writer.pages[i]
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        res = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font
        res[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = res
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td (SYN page {i} text) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    with open(path, "wb") as f:
        writer.write(f)

    r = _embed(rec_client, "full.pdf", path.read_bytes(), "application/pdf")
    assert r.status_code == 200, r.text
    rec = r.json()["extraction"]
    assert rec["status"] == "complete"
    assert rec["units_total"] == rec["units_extracted"] == 3
    assert rec["empty_locators"] == []
    stored_pages = {d.metadata.get("page") for d in _stored_docs(rec_client)}
    assert stored_pages == {0, 1, 2}


def test_embed_xlsx_sheet_receipt(rec_client, tmp_path):
    """XLSX: a workbook whose sheets all carry data reports locator_kind 'sheet'
    and complete. This used to skip whenever msoffcrypto was missing — which was
    exactly when Excel was broken; msoffcrypto-tool is now a pinned requirement
    (KI-02 SP-01.5) and the case always runs."""
    path = tmp_path / "book.xlsx"
    make_multisheet_xlsx(str(path))
    r = _embed(
        rec_client,
        "book.xlsx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert r.status_code == 200, r.text
    rec = r.json()["extraction"]
    assert rec["locator_kind"] == "sheet"
    assert rec["status"] in {"complete", "partial"}
    assert rec["units_extracted"] >= 1


# ===========================================================================
# F-ROUTES1 -- count/write parity on EVERY intake route, not just /embed
#
# `test_embed_partial_pdf_reports_partial_and_counts_match_writes` above proves
# the receipt describes the store -- for /embed. THREE routes build a receipt
# through the same `_assert_extractable_content` (/local/embed, /embed,
# /embed-upload), and a shared producer is exactly the reason not to read one
# route's result as the others': each route owns its own
# `store_data_in_vector_db` call, its own response assembly and its own
# `extraction` key, so the receipt and the rows can only be compared per route,
# at that route's own exit.
#
# Parametrized rather than looped over routes inside one test. A loop stops at
# the first failing assertion, so one route's defect would surface as "the test
# failed" with the remaining routes never exercised -- and a loop that `break`s
# or returns early on the first route it finds would pass while proving nothing
# about the other two. One test id per route means a per-route over-claim
# reddens exactly that route and leaves the others green, which is the property
# the failure control below actually measures.
# ===========================================================================


def _embed_upload(client, filename, content, content_type):
    return client.post(
        "/embed-upload",
        data={"file_id": "f-recv-upload", "entity_id": "userA"},
        files={"uploaded_file": (filename, io.BytesIO(content), content_type)},
        headers=_hdr(),
    )


def _local_embed(client, filename, content, content_type):
    """/local/embed takes a PATH inside RAG_UPLOAD_DIR, not bytes, so the same
    fixture has to be handed over a different way for this to stay the same
    comparison."""
    from app.config import RAG_UPLOAD_DIR

    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)
    path = os.path.join(RAG_UPLOAD_DIR, filename)
    with open(path, "wb") as fh:
        fh.write(content)
    try:
        return client.post(
            "/local/embed",
            json={
                "file_id": "f-recv-local",
                "filename": filename,
                "filepath": filename,
                "file_content_type": content_type,
            },
            params={"entity_id": "userA"},
            headers=_hdr(),
        )
    finally:
        # the route has finished loading the file by the time it answers
        try:
            os.remove(path)
        except OSError:
            pass


_INTAKE_ROUTES = {
    "/embed": _embed,
    "/embed-upload": _embed_upload,
    "/local/embed": _local_embed,
}


@pytest.mark.parametrize("route", sorted(_INTAKE_ROUTES))
def test_intake_route_receipt_matches_what_that_route_stored(
    route, rec_client, tmp_path
):
    """Per route: `units_extracted` must equal the number of pages that actually
    produced stored chunks, and no page the receipt calls EMPTY may hold rows.

    Both directions are asserted because they fail differently. Over-claiming is
    the dangerous one -- a unit reported extracted with no row behind it is a
    citation that resolves to nothing. Under-claiming is the quiet one -- a page
    listed empty while its rows exist tells a consumer content is missing when it
    is not.
    """
    path = tmp_path / "partial.pdf"
    make_partial_pdf(str(path))

    r = _INTAKE_ROUTES[route](
        rec_client, "partial.pdf", path.read_bytes(), "application/pdf"
    )
    assert r.status_code == 200, f"{route}: {r.text}"

    rec = r.json()["extraction"]
    stored_pages = {d.metadata.get("page") for d in _stored_docs(rec_client)}

    # PRECONDITIONS -- the fixture must be able to express the defect on THIS
    # route. If every unit extracted, or nothing reached the store, the equality
    # below would hold for reasons that have nothing to do with truthfulness.
    assert rec["locator_kind"] == "page", f"{route}: {rec}"
    assert rec["status"] == "partial", f"{route}: {rec}"
    assert stored_pages == {0, 1, 2}, f"{route}: stored {sorted(stored_pages)}"
    assert rec["empty_locators"] == [3, 4], f"{route}: {rec['empty_locators']}"

    assert rec["units_extracted"] == len(stored_pages), (
        f"{route} misreports its own write: receipt claims "
        f"{rec['units_extracted']} unit(s) extracted, the store holds rows for "
        f"{len(stored_pages)} page(s) {sorted(stored_pages)}"
    )

    leaked = sorted(set(rec["empty_locators"]) & stored_pages)
    assert not leaked, (
        f"{route}: page(s) {leaked} are reported EMPTY by the receipt but hold "
        f"stored rows"
    )

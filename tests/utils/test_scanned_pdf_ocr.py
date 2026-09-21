"""Local-first OCR for scanned PDFs (FILES-01, FS-CONTINUE-R3).

A scanned PDF is a photograph of a page: pypdf reads zero characters from it, so the
honest-but-useless answer used to be 422 "no extractable text". Richard approved
reading native text first, then BOUNDED local OCR on the pages that need it, with a
typed outcome Core can escalate on to the already-approved AWS route.

WHAT THESE TESTS ARE BUILT FROM
-------------------------------
The goals, not the code, and specifically the five fields Core asked for:

  1. per-PAGE provenance (which producer read which page)   -> test_..._provenance...
  2. coverage: yielded / attempted-and-got-nothing / never attempted, kept DISTINCT
                                                             -> test_..._never_attempted...
  3. partial and failure states that cannot read as success  -> test_sideways..., test_..._partial
  4. bounded execution with the BOUND SURFACED               -> test_page_limit..., test_time_limit...
  5. encrypted refusal still actionable, owner-password still readable
                                                             -> test_locked..., test_owner_password...

Fixtures are generated here by PIL + pypdf: a bitmap of text embedded as a JPEG on a
page with NO font and no text operators, which is structurally what a scanner produces.
No client content.

QUALITY IS MEASURED, NOT ASSERTED BY MARKER. `char_recall` is a longest-common-
subsequence ratio over alphanumerics against known ground truth. An earlier word-level
metric scored a PERFECT extraction at 0.63 because the engine joined two words -- the
metric was wrong, not the engine -- so these assert on characters recovered.

A KNOWN, MEASURED LIMITATION, deliberately left visible rather than hidden:
an UPSIDE-DOWN (180 deg) scan with no /Rotate is read at LOW recall (0.31-0.43 across
fixtures measured here and independently by review) but HIGH confidence (~0.95), with
horizontal boxes -- so neither confidence nor geometry detects it. It is
reported as recovered. `test_upside_down_scan_is_a_known_blind_spot` pins that as the
CURRENT behaviour so it cannot change silently, and it is named as a gap to Core rather
than papered over. rapidocr's `use_angle_cls` was measured and does NOT fix it.
"""

import datetime
import difflib
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from main import app
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils import ocr as ocr_module
from app.utils.document_loader import SafePyPDFLoader
from app.utils.ocr import OcrBudget, OcrCancelled, ocr_page

_SECRET = "test_key"
PDF = "application/pdf"

INVOICE = [
    "FIFTH SEASON CONSULTING",
    "Invoice 2026-0914",
    "Client: Northwind Trading Company",
    "Period: August 2026",
    "Total due: 48,250.00 EUR",
]
MEMO = [
    "Internal memorandum",
    "Subject: quarterly capacity review",
    "The consulting team reviewed utilisation",
    "across four delivery streams and found",
    "sustained demand in the retail segment.",
]
#: Rendered at a deliberately STABLE size, and getting there was a finding rather than
#: tuning. At 42px this table sat on the detector's knife-edge: pypdf re-encodes an
#: embedded JPEG (62,277 -> 62,272 bytes, mean pixel difference 0.002) and that alone
#: moved character recall from 0.98 to 0.69, losing a whole row label.
#:
#: The obvious fix -- render it bigger -- made it WORSE, not better: on a 2400x3000
#: canvas recall collapsed to 0.19 at every font size tried, because the detector
#: downscales its input and a taller page shrinks the text in detector space. Measured
#: sweep: 42px 0.69 | 48px 1.00 | 56px 1.00 | 64px 0.98, all at 1700x2200; and
#: 56/72/90px all <=0.45 at 2400x3000. So 52px at 1700x2200 sits in the middle of the
#: stable band. That non-monotonic behaviour is recorded in the operator note, because
#: it means a higher-resolution scan is not automatically a better-read one.
TABLE_ROWS = [
    [("Project", 760), ("Hours", 320), ("Rate", 320)],
    [("Retail transformation", 760), ("120", 320), ("185", 320)],
    [("Supply chain review", 760), ("64", 320), ("210", 320)],
]


# ---------------------------------------------------------------------------
# Fixture building: a real raster page, the way a scanner makes one
# ---------------------------------------------------------------------------


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render(lines=None, rows=None, size=42, rotate=0, w=1700, h=2200, margin=110):
    """Draw text onto a white bitmap -- the page a scanner would photograph."""
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    font = _font(size)
    y = margin
    if rows:
        for row in rows:
            x = margin
            for cell, width in row:
                draw.text((x, y), cell, fill="black", font=font)
                x += width
            y += int(size * 1.7)
    else:
        for line in lines or []:
            draw.text((margin, y), line, fill="black", font=font)
            y += int(size * 1.7)
    if rotate:
        image = image.rotate(rotate, expand=True)
    return image


def _image_page(writer, image, rotate_attr=None):
    """Embed `image` as a JPEG on a page carrying NO text layer."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    data = buffer.getvalue()
    width, height = image.size
    page_w, page_h = 612, 792

    xobject = DecodedStreamObject()
    xobject.set_data(data)
    xobject[NameObject("/Type")] = NameObject("/XObject")
    xobject[NameObject("/Subtype")] = NameObject("/Image")
    xobject[NameObject("/Width")] = NumberObject(width)
    xobject[NameObject("/Height")] = NumberObject(height)
    xobject[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    xobject[NameObject("/BitsPerComponent")] = NumberObject(8)
    xobject[NameObject("/Filter")] = NameObject("/DCTDecode")
    reference = writer._add_object(xobject)

    page = writer.add_blank_page(width=page_w, height=page_h)
    named = DictionaryObject()
    named[NameObject("/Im0")] = reference
    resources = DictionaryObject()
    resources[NameObject("/XObject")] = named
    page[NameObject("/Resources")] = resources
    content = DecodedStreamObject()
    content.set_data(f"q {page_w} 0 0 {page_h} 0 0 cm /Im0 Do Q".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)
    if rotate_attr:
        page[NameObject("/Rotate")] = NumberObject(rotate_attr)
    return page


def _text_page(writer, lines, size=12):
    """A page with a real text layer -- what pypdf can already read."""
    page = writer.add_blank_page(width=612, height=792)
    operators = ["BT /F1 %d Tf" % size]
    y = 720
    for line in lines:
        operators.append(f"1 0 0 1 72 {y} Tm ({line}) Tj")
        y -= int(size * 1.6)
    operators.append("ET")
    stream = DecodedStreamObject()
    stream.set_data(" ".join(operators).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources
    return page


def _text_and_image_page(writer, text_lines, image, text_size=12):
    """One page carrying BOTH a real text layer AND an embedded image.

    This is what a page looks like when a small caption/header/footer is typed over
    a scanned body: pypdf reads the text operators, so `page_content` is non-empty and
    the OCR path (which only fires on an EMPTY page) never looks at the image. Used to
    pin the text-first coverage limitation, not to exercise OCR.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    data = buffer.getvalue()
    width, height = image.size
    page_w, page_h = 612, 792

    xobject = DecodedStreamObject()
    xobject.set_data(data)
    xobject[NameObject("/Type")] = NameObject("/XObject")
    xobject[NameObject("/Subtype")] = NameObject("/Image")
    xobject[NameObject("/Width")] = NumberObject(width)
    xobject[NameObject("/Height")] = NumberObject(height)
    xobject[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    xobject[NameObject("/BitsPerComponent")] = NumberObject(8)
    xobject[NameObject("/Filter")] = NameObject("/DCTDecode")
    reference = writer._add_object(xobject)

    page = writer.add_blank_page(width=page_w, height=page_h)
    named = DictionaryObject()
    named[NameObject("/Im0")] = reference

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)

    resources = DictionaryObject()
    resources[NameObject("/XObject")] = named
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources

    operators = ["q %d 0 0 %d 0 0 cm /Im0 Do Q" % (page_w, page_h), "BT /F1 %d Tf" % text_size]
    y = 760
    for line in text_lines:
        operators.append(f"1 0 0 1 72 {y} Tm ({line}) Tj")
        y -= int(text_size * 1.6)
    operators.append("ET")
    content = DecodedStreamObject()
    content.set_data(" ".join(operators).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)
    return page


def _bytes(writer):
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def scanned_pdf(lines=None, rows=None, rotate=0, rotate_attr=None, pages=1, size=42,
                w=1700, h=2200):
    writer = PdfWriter()
    for _ in range(pages):
        _image_page(
            writer,
            _render(lines, rows, size=size, rotate=rotate, w=w, h=h),
            rotate_attr=rotate_attr,
        )
    return _bytes(writer)


# ---------------------------------------------------------------------------
# Measuring extraction quality honestly
# ---------------------------------------------------------------------------


def _normalise(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def char_recall(got, truth_lines):
    """Share of the ground-truth CHARACTERS recovered, in order.

    Character-level on purpose: a word-level metric scored a perfect extraction at
    0.63 merely because the engine joined "quarterly capacity" into one token.
    """
    truth = _normalise(" ".join(truth_lines))
    matcher = difflib.SequenceMatcher(None, truth, _normalise(got), autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(truth)


# ---------------------------------------------------------------------------
# Route harness
# ---------------------------------------------------------------------------


def _hdr():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "testuser",
        "tid": "tenantA",
        "ent": ["userA"],
        "act": ["write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture()
def client(monkeypatch):
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    written = []

    async def record(self, docs, ids=None, executor=None):
        written.append(list(docs))
        return ids or ["id1"]

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", record)
    test_client = TestClient(app)
    test_client.written = written  # type: ignore[attr-defined]
    return test_client


def _embed(client, name, data):
    return client.post(
        "/embed",
        data={"file_id": "ocr-" + name, "entity_id": "userA"},
        files={"file": (name, io.BytesIO(data), PDF)},
        headers=_hdr(),
    )


def _stored(client):
    return "\n".join(d.page_content for batch in client.written for d in batch)


def _receipt(response):
    body = response.json()
    detail = body.get("detail", body)
    return (detail or {}).get("extraction", {})


# ===========================================================================
# The capability: a scan that used to fail now actually reads
# ===========================================================================


def test_a_genuine_scan_is_read_and_stored(client):
    """The user-visible outcome: a scanned invoice used to be refused with 422 and
    zero rows. It must now be ingested, with the text actually reaching the store."""
    response = _embed(client, "scan.pdf", scanned_pdf(INVOICE))

    assert response.status_code == 200, response.text
    stored = _stored(client)
    assert stored, "a scanned PDF produced no rows"
    recovered = char_recall(stored, INVOICE)
    assert recovered >= 0.9, f"only {recovered:.2f} of the page's characters were recovered: {stored!r}"

    receipt = _receipt(response)
    assert receipt["ocr"]["pages_recovered"] == 1
    assert receipt["ocr"]["engine"] == "rapidocr-onnxruntime"
    assert receipt["escalation"]["recommended"] is False


def test_a_scanned_table_keeps_its_values(client):
    """Tables are the case where a wrong number is worse than no number."""
    response = _embed(client, "table.pdf", scanned_pdf(rows=TABLE_ROWS, size=52))

    assert response.status_code == 200, response.text
    stored = _stored(client)
    flat = [cell for row in TABLE_ROWS for cell, _ in row]
    assert char_recall(stored, flat) >= 0.9, stored
    for value in ("120", "185", "210"):
        assert value in stored, f"table value {value} was lost: {stored!r}"


def test_the_pdfs_own_rotation_is_honoured(client):
    """A scanner that records /Rotate must be read the right way up. Measured, this
    alone is the difference between 0.31 and 0.99 character recall."""
    for angle in (90, 180, 270):
        client.written.clear()
        data = scanned_pdf(INVOICE, rotate=angle, rotate_attr=angle)
        response = _embed(client, f"rot{angle}.pdf", data)
        assert response.status_code == 200, response.text
        recovered = char_recall(_stored(client), INVOICE)
        assert recovered >= 0.9, f"/Rotate={angle} recovered only {recovered:.2f}"


# ===========================================================================
# Field 1: per-page provenance. Field 3: no duplicate text for mixed documents.
# ===========================================================================


def test_native_pages_are_never_ocrd_and_text_is_never_duplicated(client):
    """A mixed document: page 0 has a real text layer, page 1 is a scan.

    Two things must hold at once. The native page must be reported as native and
    reach the store EXACTLY ONCE -- OCR-ing a page that already has text is how this
    design would produce duplicate content. And the scan must still be read.
    """
    writer = PdfWriter()
    _text_page(writer, MEMO)
    _image_page(writer, _render(INVOICE))
    response = _embed(client, "mixed.pdf", _bytes(writer))

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    assert receipt["text_sources"] == {"native": [0], "ocr": [1]}, receipt["text_sources"]
    # The native page was never handed to the engine.
    assert receipt["ocr"]["pages_attempted"] == 1

    stored = _stored(client)
    assert char_recall(stored, MEMO) >= 0.9
    assert char_recall(stored, INVOICE) >= 0.9
    # The native page's distinctive line must appear once, not twice.
    assert stored.count("Internal memorandum") == 1, stored


def test_provenance_reaches_the_index_not_just_the_receipt(client):
    """Provenance has to survive INTO the stored rows, because that is where retrieval
    reads it -- a receipt is thrown away after the upload call.

    This test exists because mutation testing found the gap: deleting the loader's
    provenance marking left the receipt assertion GREEN, since the receipt can fall
    back to inferring `native` from "this unit has content". The fallback is right for
    formats with no per-page producer (workbooks, decks), but it meant nothing was
    actually proving the PDF path labels its own pages. The stored chunk cannot be
    faked by that fallback, so this is the assertion that bites.
    """
    writer = PdfWriter()
    _text_page(writer, MEMO)
    _image_page(writer, _render(INVOICE))
    response = _embed(client, "mixed.pdf", _bytes(writer))
    assert response.status_code == 200, response.text

    by_source = {}
    for batch in client.written:
        for document in batch:
            by_source.setdefault(document.metadata.get("text_source"), []).append(
                document.metadata.get("page")
            )

    assert set(by_source) == {"native", "ocr"}, (
        f"every stored chunk must carry its producer; got {sorted(by_source)}"
    )
    assert set(by_source["native"]) == {0}
    assert set(by_source["ocr"]) == {1}
    # An OCR chunk also carries the engine's own confidence, for retrieval-time triage.
    ocr_chunks = [
        d for b in client.written for d in b if d.metadata.get("text_source") == "ocr"
    ]
    assert all(0.0 < d.metadata["ocr_confidence"] <= 1.0 for d in ocr_chunks)


def test_a_native_pdf_costs_nothing_and_reports_no_ocr(client):
    """Regression guard on the whole existing PDF path: a text PDF must behave
    exactly as it did before OCR existed -- same 200, same text, and no `ocr` block
    at all, so every existing receipt keeps its shape."""
    writer = PdfWriter()
    _text_page(writer, INVOICE)
    started = time.monotonic()
    response = _embed(client, "native.pdf", _bytes(writer))
    elapsed = time.monotonic() - started

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    assert "ocr" not in receipt, "a native PDF must not report an OCR block"
    assert "escalation" not in receipt
    assert receipt["status"] == "complete"
    assert char_recall(_stored(client), INVOICE) >= 0.95
    assert elapsed < 5, f"a native PDF took {elapsed:.1f}s; OCR is being run on it"


# ===========================================================================
# Field 3: partial and failure states that cannot be mistaken for success
# ===========================================================================


def test_sideways_scan_is_never_reported_as_complete(client):
    """THE HONESTY GUARD, and the defect the fixture corpus actually found.

    A page rotated 90 degrees with no /Rotate to say so comes back with only a fraction of
    its characters (0.37-0.51 measured across fixtures) -- but at HIGH mean confidence
    (0.89-0.96), because confidence averages the
    lines that WERE detected and is blind to everything missed. The first version of
    this feature reported that page as `complete` with no escalation: nonempty text
    presented as success, which is exactly the failure Core named.

    The signal is geometric, not a quality score: on a sideways page the detected text
    boxes run vertically (measured: tall-box ratio 1.00, versus 0.00 on every upright
    page in the corpus).
    """
    response = _embed(client, "sideways.pdf", scanned_pdf(INVOICE, rotate=90))

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    assert receipt["ocr"]["pages_orientation_suspect"] == 1
    # ...and NOT also under low confidence. Review found the first version counting this
    # page in BOTH buckets: one weak page, two increments, and a false label besides,
    # because a sideways page comes back CONFIDENT -- which is the entire reason the
    # geometric signal had to exist. A counter that double-counts is a counter that lies.
    assert receipt["ocr"]["pages_low_confidence"] == 0, receipt["ocr"]
    assert receipt["ocr"]["mean_confidence"] > 0.8, (
        "this page must be recorded as HIGH confidence; if it were low, confidence alone "
        "would have caught it and the geometric signal would be unnecessary"
    )
    assert receipt["escalation"]["recommended"] is True
    assert receipt["escalation"]["reason"] == "ocr_orientation_suspect"
    assert receipt["escalation"]["locators"] == [0]
    assert receipt["status"] == "partial", (
        "a page that still needs a better reader must never read as `complete` on the "
        "field consumers already check"
    )


def test_upside_down_scan_is_a_known_blind_spot(client):
    """PINNED, NOT CELEBRATED. An upside-down scan is read at ~0.31 recall with 0.95
    confidence and horizontal boxes, so nothing local detects it and it IS reported as
    recovered. rapidocr's `use_angle_cls` was measured and does not fix it.

    This test exists so the limitation is visible and cannot change silently -- if a
    later change starts detecting it, this test fails and the gap note gets updated.
    """
    response = _embed(client, "upsidedown.pdf", scanned_pdf(INVOICE, rotate=180))

    assert response.status_code == 200, response.text
    recovered = char_recall(_stored(client), INVOICE)
    assert recovered < 0.75, (
        f"an upside-down scan now recovers {recovered:.2f} of its characters -- the "
        "known blind spot may have been fixed; update the gap note and Core's contract"
    )
    assert _receipt(response)["escalation"]["recommended"] is False, (
        "still undetected locally -- this is the documented gap, reported to Core"
    )


def test_a_text_layer_page_with_an_unread_image_is_a_known_coverage_gap(client):
    """PINNED, NOT CELEBRATED. The text-first coverage limitation, measured.

    OCR fires only on a page whose `page_content` is empty. So a page that carries a
    real text layer -- even a one-line header or footer -- is stamped `native` and its
    embedded image is NEVER read, however much text that image holds. Here the text
    layer is a single header line and the image holds the whole INVOICE. Only the header
    reaches the store, and the receipt now reports `status: partial` with `image_ocr:
    not_attempted` and no `ocr` block: since #66 the gap is DISCLOSED rather than hidden
    behind `complete`. The gap itself is unchanged -- the image is still never read --
    which is precisely the case A07 warns about: text on the page is not proof the page
    was read.

    Whether to OCR images on pages that already have a text layer is an OPEN PRODUCT
    DECISION (cost/latency), NOT decided here: measured cost is ~1.3 s/page steady-state
    in this image (first page ~4.5 s incl. engine warm-up). See the F05 coverage receipt.

    This test exists so the gap is visible and cannot change silently. If a later change
    starts OCR-ing text-layer pages (or flags the unread image), the image text will
    reach the store / an `ocr` block will appear and this test will fail -- forcing a
    conscious update of the contract and Richard's open question rather than a silent
    semantic drift.
    """
    writer = PdfWriter()
    _text_and_image_page(writer, ["Page 1 header"], _render(INVOICE), text_size=12)
    response = _embed(client, "textlayer_over_scan.pdf", _bytes(writer))

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    # The page declares the gap on the field consumers check...
    assert receipt["status"] == "partial", receipt
    assert receipt["coverage"]["image_ocr"] == "not_attempted", receipt
    # ...because the text layer made it `native`, so OCR never looked at the image.
    assert "ocr" not in receipt, "the image must not have been OCR'd on a text-layer page"
    assert "escalation" not in receipt
    assert receipt["text_sources"] == {"native": [0]}, receipt.get("text_sources")

    stored = _stored(client)
    assert "Page 1 header" in stored, "the text layer must be stored"
    # THE GAP: the image's text is absent from the store. If this recall climbs, the gap
    # is closing (someone started reading text-layer images) and the contract/question
    # above must be revisited.
    assert char_recall(stored, INVOICE) < 0.3, (
        f"the image's text is now reaching the store (recall {char_recall(stored, INVOICE):.2f}) "
        "-- the text-first coverage gap may have been closed; update the F05 receipt and "
        "Core's contract before relaxing this guard"
    )


def test_an_unreadable_page_yields_nothing_and_asks_for_escalation(client):
    """A page OCR genuinely cannot read must store nothing and say so specifically --
    `ocr_no_text`, not the generic `empty` a blank page gets."""
    noise = Image.new("RGB", (1700, 2200), "white")
    pixels = noise.load()
    import random

    random.seed(3)
    for _ in range(400_000):
        x, y = random.randrange(1700), random.randrange(2200)
        pixels[x, y] = (random.randrange(256),) * 3
    writer = PdfWriter()
    _image_page(writer, noise)

    response = _embed(client, "noise.pdf", _bytes(writer))

    assert response.status_code == 422, response.text
    assert client.written == [], "an unreadable page must write nothing"
    receipt = _receipt(response)
    assert receipt["status"] == "empty"
    assert receipt["ocr"]["pages_no_text"] == 1
    assert receipt["escalation"]["recommended"] is True
    assert receipt["reasons"] == [{"locator": 0, "reason": "ocr_no_text"}]


def test_a_page_with_nothing_to_read_is_not_counted_as_attempted(client):
    """`pages_attempted` must mean "the engine looked at this page".

    Review found it being set before the image check, so a page with no embedded image
    at all counted as attempted -- the field then meant "we considered this page", which
    is not what its docstring says and not what Core is reading it for.
    """
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    response = _embed(client, "blank.pdf", _bytes(writer))

    receipt = _receipt(response)
    assert receipt["ocr"]["pages_attempted"] == 0, receipt["ocr"]
    assert receipt["ocr"]["pages_no_text"] == 0, "nothing was read, so nothing came back empty"
    assert receipt["escalation"]["recommended"] is False


def test_a_blank_page_does_not_invent_an_escalation(client):
    """A genuinely blank page has no image to read. We cannot tell that apart from a
    vector-only page HERE, so we must not claim it needs a better reader -- that would
    be manufacturing a fact. It keeps the plain `empty` it reported before OCR."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    response = _embed(client, "blank.pdf", _bytes(writer))

    assert response.status_code == 422, response.text
    receipt = _receipt(response)
    assert receipt["status"] == "empty"
    assert receipt["reasons"] == [{"locator": 0, "reason": "empty"}]
    assert receipt["escalation"]["recommended"] is False


# ===========================================================================
# Field 2: coverage -- "read it and got nothing" vs "never looked"
# ===========================================================================


def test_never_attempted_pages_are_not_counted_as_attempted(client, monkeypatch):
    """Core called this the distinction that gets collapsed by accident, and it is:
    a page skipped because the page budget was already spent is NOT a page we read and
    found empty. They mean different things to a user and they are counted separately.
    """
    monkeypatch.setattr("app.utils.ocr.PDF_OCR_MAX_PAGES", 1)

    response = _embed(client, "three.pdf", scanned_pdf(INVOICE, pages=3))

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    ocr = receipt["ocr"]
    assert ocr["pages_attempted"] == 1, ocr
    assert ocr["pages_not_attempted"] == 2, ocr
    assert ocr["pages_no_text"] == 0, "a page we never looked at is not a page with no text"
    assert ocr["stopped_reason"] == "page_limit", "the bound that stopped the work must be named"
    assert receipt["status"] == "partial"
    assert receipt["escalation"]["recommended"] is True
    assert receipt["escalation"]["locators"] == [1, 2]


def test_the_time_bound_is_surfaced_not_silent(client, monkeypatch):
    """A cap that truncates silently is worse than a refusal."""
    monkeypatch.setattr("app.utils.ocr.PDF_OCR_TIME_BUDGET_SECONDS", 0.0)

    response = _embed(client, "slow.pdf", scanned_pdf(INVOICE, pages=2))

    assert response.status_code == 422, response.text
    receipt = _receipt(response)
    assert receipt["ocr"]["stopped_reason"] == "time_limit"
    assert receipt["ocr"]["pages_attempted"] == 0
    assert receipt["ocr"]["pages_not_attempted"] == 2
    assert receipt["escalation"]["recommended"] is True


# ===========================================================================
# Field 5: the encrypted contract is unchanged by any of this
# ===========================================================================


def test_a_locked_scan_is_still_refused_and_never_ocrd(client):
    """A password-protected PDF must keep its actionable 422 -- OCR must not become a
    way to read around a password."""
    writer = PdfWriter()
    _image_page(writer, _render(INVOICE))
    writer.encrypt(user_password="userpw")

    response = _embed(client, "locked.pdf", _bytes(writer))

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["extraction"]["verdict"] == "encrypted"
    assert "without a password" in detail["message"]
    assert client.written == []


def test_an_owner_password_scan_is_read_like_any_other_scan(client):
    """An owner-password PDF is READABLE -- it only restricts printing and copying.
    It must reach OCR like any other scan, not be refused."""
    writer = PdfWriter()
    _image_page(writer, _render(MEMO))
    writer.encrypt(user_password="", owner_password="ownerpw")

    response = _embed(client, "ownerpw.pdf", _bytes(writer))

    assert response.status_code == 200, response.text
    assert char_recall(_stored(client), MEMO) >= 0.9


# ===========================================================================
# Degradation: OCR must never become a new way to fail
# ===========================================================================


def test_an_unavailable_engine_degrades_honestly(client, monkeypatch):
    """If OCR cannot load, a scan must fail exactly as it did before OCR existed --
    an honest 422 with zero rows, never a 5xx -- and must ask for escalation."""
    monkeypatch.setattr(ocr_module, "get_engine", lambda: None)

    response = _embed(client, "scan.pdf", scanned_pdf(INVOICE))

    assert response.status_code == 422, response.text
    assert client.written == []
    receipt = _receipt(response)
    assert receipt["ocr"]["pages_attempted"] == 0
    assert receipt["escalation"]["reason"] == "ocr_unavailable"
    assert receipt["escalation"]["recommended"] is True


def test_ocr_can_be_switched_off_entirely(client, monkeypatch):
    """The operator kill switch. With OCR disabled the service must behave exactly as
    the pre-OCR build did: 422, zero rows, and no OCR block in the receipt."""
    monkeypatch.setattr("app.utils.ocr.PDF_OCR_ENABLED", False)

    response = _embed(client, "scan.pdf", scanned_pdf(INVOICE))

    assert response.status_code == 422, response.text
    assert client.written == []
    receipt = _receipt(response)
    assert "ocr" not in receipt
    assert receipt["reasons"] == [{"locator": 0, "reason": "empty"}]


def test_the_kill_switch_leaves_no_residue_on_a_native_pdf(client, monkeypatch):
    """The switch has to be TOTAL, including on the format the feature touches.

    Review found provenance being stamped on native pages unconditionally, so a native
    PDF gained a `text_sources` block even with OCR disabled -- which made the
    "byte-identical to the pre-OCR build" claim false exactly where the kill switch is
    meant to be complete. The switch is what an operator reaches for when something is
    wrong, so 'almost off' is not a state it may have.
    """
    monkeypatch.setattr("app.utils.document_loader.PDF_OCR_ENABLED", False)
    monkeypatch.setattr("app.utils.ocr.PDF_OCR_ENABLED", False)

    writer = PdfWriter()
    _text_page(writer, INVOICE)
    response = _embed(client, "native.pdf", _bytes(writer))

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    for added in ("ocr", "escalation", "text_sources"):
        assert added not in receipt, f"{added} survived the kill switch: {receipt}"
    assert char_recall(_stored(client), INVOICE) >= 0.95, "the PDF must still ingest"


def test_a_native_pdf_carries_provenance_while_ocr_is_on(client):
    """The other side of the switch: with OCR enabled a native PDF DOES report its
    provenance, because Core reads `text_source` at retrieval time across the whole
    corpus and a missing producer would read as an unknown one."""
    writer = PdfWriter()
    _text_page(writer, INVOICE)
    response = _embed(client, "native.pdf", _bytes(writer))

    assert response.status_code == 200, response.text
    receipt = _receipt(response)
    assert receipt["text_sources"] == {"native": [0]}
    assert "ocr" not in receipt, "no OCR ran, so no OCR block"
    assert all(
        d.metadata.get("text_source") == "native"
        for batch in client.written
        for d in batch
    )


def test_one_unreadable_image_does_not_lose_the_rest_of_the_page(client, monkeypatch):
    """A page can carry several images. One that cannot be decoded must not discard
    the text recovered from the others."""
    calls = {"n": 0}
    real = ocr_module._to_array

    def flaky(data, rotation, budget, notes):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("cannot decode this one")
        return real(data, rotation, budget, notes)

    monkeypatch.setattr(ocr_module, "_to_array", flaky)

    writer = PdfWriter()
    page = _image_page(writer, _render(INVOICE))
    # a second image on the same page
    second = _render(MEMO)
    buffer = io.BytesIO()
    second.save(buffer, format="JPEG", quality=80)
    extra = DecodedStreamObject()
    extra.set_data(buffer.getvalue())
    extra[NameObject("/Type")] = NameObject("/XObject")
    extra[NameObject("/Subtype")] = NameObject("/Image")
    extra[NameObject("/Width")] = NumberObject(second.width)
    extra[NameObject("/Height")] = NumberObject(second.height)
    extra[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    extra[NameObject("/BitsPerComponent")] = NumberObject(8)
    extra[NameObject("/Filter")] = NameObject("/DCTDecode")
    page[NameObject("/Resources")][NameObject("/XObject")][NameObject("/Im1")] = (
        writer._add_object(extra)
    )

    response = _embed(client, "twoimages.pdf", _bytes(writer))

    assert response.status_code == 200, response.text
    assert calls["n"] >= 2, "the second image was never attempted"
    assert char_recall(_stored(client), MEMO) >= 0.9


# ===========================================================================
# Cancellation: stop working, do not merely stop waiting
# ===========================================================================


def test_cancellation_stops_the_work_between_pages(tmp_path):
    """`run_in_executor` cancels the FUTURE, never the worker thread -- so without a
    cooperative flag a disconnected client leaves a 50-page OCR burning CPU for
    nobody. This proves the loader stops rather than running the document out.
    """
    path = tmp_path / "many.pdf"
    path.write_bytes(scanned_pdf(INVOICE, pages=6))

    seen = {"pages": 0}
    real_ocr_page = ocr_page

    def counting(page, budget):
        seen["pages"] += 1
        return real_ocr_page(page, budget)

    stop_after = 2

    budget = OcrBudget(should_stop=lambda: seen["pages"] >= stop_after)
    loader = SafePyPDFLoader(str(path), ocr_budget=budget)

    import app.utils.ocr as module

    original = module.ocr_page
    module.ocr_page = counting
    try:
        with pytest.raises(OcrCancelled):
            list(loader.lazy_load())
    finally:
        module.ocr_page = original

    assert seen["pages"] == stop_after, (
        f"OCR kept going after cancellation: {seen['pages']} pages processed"
    )
    assert budget.stopped_reason == "cancelled"


# ===========================================================================
# Units: the pieces the route tests rely on
# ===========================================================================


def test_a_tall_text_box_is_what_sideways_looks_like():
    """The orientation signal is geometry, not judgement."""
    wide = [[0, 0], [100, 0], [100, 20], [0, 20]]
    tall = [[0, 0], [20, 0], [20, 100], [0, 100]]
    assert ocr_module._is_tall(tall) is True
    assert ocr_module._is_tall(wide) is False
    assert ocr_module._is_tall("not a box") is False


def test_low_confidence_is_a_report_not_a_discard():
    """rag_api reports weak text; it does not decide the document's fate. Text is
    flagged, never thrown away -- judging coverage is Core's boundary."""
    assert ocr_module.is_low_confidence("a" * 100, 0.2) is True
    assert ocr_module.is_low_confidence("short", 0.99) is True
    assert ocr_module.is_low_confidence("a" * 100, 0.95) is False
    # No text at all is a DIFFERENT outcome and is reported separately.
    assert ocr_module.is_low_confidence("   ", 0.0) is False


def test_the_budget_names_the_bound_that_ran_out():
    budget = OcrBudget(max_pages=2, time_budget_seconds=60)
    assert budget.may_start_page() is True
    assert budget.may_start_page() is True
    assert budget.may_start_page() is False
    assert budget.stopped_reason == "page_limit"

    spent = OcrBudget(max_pages=10, time_budget_seconds=0)
    assert spent.may_start_page() is False
    assert spent.stopped_reason == "time_limit"

    cancelled = OcrBudget(should_stop=lambda: True)
    with pytest.raises(OcrCancelled):
        cancelled.may_start_page()
    assert cancelled.stopped_reason == "cancelled"

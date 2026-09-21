"""A mixed text/image page is never reported fully covered (FILES-01 D-F05).

Richard's ruling (2026-09-21): a text layer alone does not establish complete coverage of
a page containing images. Record text extraction and image/OCR processing SEPARATELY; if
image content was not processed or its coverage is unknown, disclose it; do not claim
full-page coverage.

Before: a page with any native text was stamped `native` and its embedded image was never
OCR'd, yet the receipt reported `status: complete` with no signal that the image was
unread (measured in the F05 audit: header-only text layer + full-invoice image, image
recall ~0.06, status complete). This pins the honest behaviour and its negative control.

The fixture is a single page that carries BOTH a real text layer (a header line pypdf
reads) AND an embedded image rendering text the layer lacks -- so a `complete` here would
be exactly the over-claim the ruling forbids.
"""

import io

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from tests.utils.test_scanned_pdf_ocr import (  # reuse the real harness
    INVOICE,
    _bytes,
    _embed,
    _receipt,
    _render,
    client,  # noqa: F401  (pytest fixture)
)


def _text_and_image_page(writer, header_lines, image):
    """One page with a text layer (header_lines) AND an embedded JPEG (image)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    data = buffer.getvalue()
    w, h = image.size
    page_w, page_h = 612, 792

    xobject = DecodedStreamObject()
    xobject.set_data(data)
    xobject[NameObject("/Type")] = NameObject("/XObject")
    xobject[NameObject("/Subtype")] = NameObject("/Image")
    xobject[NameObject("/Width")] = NumberObject(w)
    xobject[NameObject("/Height")] = NumberObject(h)
    xobject[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    xobject[NameObject("/BitsPerComponent")] = NumberObject(8)
    xobject[NameObject("/Filter")] = NameObject("/DCTDecode")
    img_ref = writer._add_object(xobject)

    page = writer.add_blank_page(width=page_w, height=page_h)

    # Draw the image (lower half) then the header text (top).
    ops = [f"q {page_w} 0 0 {page_h // 2} 0 0 cm /Im0 Do Q", "BT /F1 12 Tf"]
    y = 760
    for line in header_lines:
        ops.append(f"1 0 0 1 72 {y} Tm ({line}) Tj")
        y -= 20
    ops.append("ET")
    content = DecodedStreamObject()
    content.set_data(" ".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    named = DictionaryObject()
    named[NameObject("/Im0")] = img_ref
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    resources[NameObject("/XObject")] = named
    page[NameObject("/Resources")] = resources
    return page


def _mixed_pdf():
    writer = PdfWriter()
    _text_and_image_page(writer, ["Invoice summary (see attached scan)"],
                         _render(INVOICE))  # image carries the invoice body text
    return _bytes(writer)


def test_a_text_layer_page_with_an_unread_image_is_not_reported_complete(client):
    """OCR ON. The header text extracts, but the image body was never OCR'd (native won),
    so coverage of the image is not established and the receipt must not say `complete`."""
    r = _embed(client, "mixed.pdf", _mixed_pdf())
    assert r.status_code == 200, r.text
    receipt = _receipt(r)
    assert receipt["status"] != "complete", receipt
    assert "coverage" in receipt, receipt
    assert receipt["coverage"]["text"] == "complete", receipt      # the text layer WAS read
    assert receipt["coverage"]["image_ocr"] == "not_attempted", receipt


def test_the_same_page_is_not_reported_complete_with_OCR_DISABLED(client, monkeypatch):
    """NEGATIVE CONTROL (the card's requirement): turning OCR off must not make an unread
    image read as fully covered. Coverage disclosure is a fact about what was read, not an
    artefact of the OCR engine having run."""
    monkeypatch.setattr("app.utils.document_loader.PDF_OCR_ENABLED", False)
    monkeypatch.setattr("app.utils.ocr.PDF_OCR_ENABLED", False)
    r = _embed(client, "mixed.pdf", _mixed_pdf())
    assert r.status_code == 200, r.text
    receipt = _receipt(r)
    assert receipt["status"] != "complete", receipt
    assert receipt["coverage"]["image_ocr"] == "not_attempted", receipt
    # Kill switch still total for OCR OUTPUT: no ocr/text_sources block appears.
    for ocr_block in ("ocr", "escalation", "text_sources"):
        assert ocr_block not in receipt, (ocr_block, receipt)


def test_a_pure_text_page_with_no_image_stays_complete_and_carries_no_coverage_block(client):
    """The disclosure must not fire on the normal case: an imageless native PDF keeps its
    exact previous shape (no coverage block) and reads complete."""
    from tests.utils.test_scanned_pdf_ocr import _text_page
    writer = PdfWriter()
    _text_page(writer, INVOICE)
    r = _embed(client, "native.pdf", _bytes(writer))
    assert r.status_code == 200, r.text
    receipt = _receipt(r)
    assert receipt["status"] == "complete", receipt
    assert "coverage" not in receipt, receipt

"""F05 EXTRACTION: read the image on a text-layer page when the operator opts in.

#66 disclosed the gap (a text-layer page's image is `not_attempted`). This is the other
half of the master-plan F05 outcome: with PDF_OCR_MIXED_PAGE on, the embedded image is
OCR'd through the SAME local adapter and budget as a scanned page, and text it holds that
the layer lacks is stored as a sibling chunk citing the same page. Default OFF, so a native
PDF is untouched unless an operator opts in; paid fallback stays held.

Uses the real-tesseract `client` fixture from test_scanned_pdf_ocr (its image OCRs the
rendered invoice, as the scanned-PDF tests already rely on), and the mixed-page builder
from test_mixed_page_coverage (a real text layer + an image the layer does not transcribe).
"""

import io

from pypdf import PdfWriter

from app.utils import document_loader
from tests.utils.test_scanned_pdf_ocr import (
    INVOICE,
    _bytes,
    _embed,
    _receipt,
    _render,
    _stored,
    char_recall,
    client,  # noqa: F401  (pytest fixture, real tesseract + write capture)
)
from tests.utils.test_mixed_page_coverage import _text_and_image_page

HEADER = "Invoice summary -- see the attached scan below."


def _mixed_pdf():
    """One page: a typed header (text layer) + the full INVOICE rendered as an image."""
    writer = PdfWriter()
    _text_and_image_page(writer, [HEADER], _render(INVOICE))
    return _bytes(writer)


def test_off_by_default_the_image_is_not_read(client):
    """Control / kill-switch: with the flag off (default), the image is never OCR'd -- only
    the header layer is stored, and coverage stays the #66 disclosure `not_attempted`."""
    r = _embed(client, "mixed.pdf", _mixed_pdf())
    assert r.status_code == 200, r.text
    stored = _stored(client)
    assert "Invoice summary" in stored, "the text layer must still be stored"
    assert char_recall(stored, INVOICE) < 0.5, "the image body must NOT be read when off"
    assert _receipt(r)["coverage"]["image_ocr"] == "not_attempted"


def test_on_the_image_body_the_layer_lacks_becomes_searchable(client, monkeypatch):
    monkeypatch.setattr(document_loader, "PDF_OCR_MIXED_PAGE", True)
    r = _embed(client, "mixed.pdf", _mixed_pdf())
    assert r.status_code == 200, r.text
    stored = _stored(client)
    assert "Invoice summary" in stored, "the text layer is still stored"
    # The invoice body -- which the header layer does NOT contain -- is now read from the image.
    assert char_recall(stored, INVOICE) >= 0.9, f"image body not extracted: recall too low"
    # Coverage flips: the image was actually read.
    assert _receipt(r)["coverage"]["image_ocr"] == "attempted"
    # The image text is a distinct chunk, provenance ocr, citing the same page.
    ocr_chunks = [d for b in client.written for d in b
                  if d.metadata.get("mixed_page_image") is True]
    assert ocr_chunks, "the image text must be stored as its own chunk"
    assert all(d.metadata.get("text_source") == "ocr" for d in ocr_chunks)
    assert all(d.metadata.get("page") == 0 for d in ocr_chunks), "sibling must cite the page"


def test_on_a_duplicate_image_is_not_double_stored(client, monkeypatch):
    """If the image renders the SAME text as the layer, OCR adds nothing new -- the page is
    not double-stored, and coverage is still `attempted` (it WAS read)."""
    monkeypatch.setattr(document_loader, "PDF_OCR_MIXED_PAGE", True)
    writer = PdfWriter()
    # Header lines AND image are the same INVOICE text.
    _text_and_image_page(writer, INVOICE, _render(INVOICE))
    r = _embed(client, "dup.pdf", _bytes(writer))
    assert r.status_code == 200, r.text
    dup_chunks = [d for b in client.written for d in b
                  if d.metadata.get("mixed_page_image") is True]
    assert dup_chunks == [], "an image duplicating the layer must not be stored again"
    assert _receipt(r)["coverage"]["image_ocr"] == "attempted"


def test_on_an_imageless_native_page_is_untouched(client, monkeypatch):
    """No image -> nothing to OCR even with the flag on; no coverage block, stays complete."""
    monkeypatch.setattr(document_loader, "PDF_OCR_MIXED_PAGE", True)
    from tests.utils.test_scanned_pdf_ocr import _text_page
    writer = PdfWriter()
    _text_page(writer, INVOICE)
    r = _embed(client, "native.pdf", _bytes(writer))
    assert r.status_code == 200, r.text
    receipt = _receipt(r)
    assert receipt["status"] == "complete", receipt
    assert "coverage" not in receipt, receipt

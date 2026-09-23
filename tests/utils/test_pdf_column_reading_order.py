"""PDF multi-column reading order + running header/footer de-duplication (E2).

WHAT E2 ADDS. pypdf reads a two-column page in stream / row order, so a two-column
page comes out with the columns interleaved (right-column text spliced between
left-column lines) -- wrong reading order for retrieval and citation. And a running
header/footer repeated on every page is extracted on every page, inflating the
index with boilerplate. E2 adds a NATIVE-PATH-ONLY layout pass to SafePyPDFLoader
that reorders each native page into column reading order (each column top-to-bottom,
columns left-to-right) and drops running headers/footers.

GROUND TRUTH is the AUTHOR order: in a two-column document the author intends the
left column read top-to-bottom, then the right column. The fixtures are hand-built
PDFs whose every text run's x/y is set with explicit PDF operators
(`BT/Tf/Tm/Tj/ET`) through pypdf -- NO reportlab, no new dependency -- so the
ground-truth reading order and the header/footer bands are exact and declared.

THE OCR BOUNDARY. The pass runs strictly UPSTREAM of `_with_ocr`: it only ever sees
pypdf native text, a scanned (empty-text) page passes through untouched, and OCR
output is produced downstream and never reordered. That boundary is proven here,
not asserted (test_ocr_output_is_not_reordered_by_the_layout_pass_SYNTHETIC and
test_a_scanned_page_reaches_the_layout_pass_empty_SYNTHETIC).

All fixtures are SYNTHETIC and generated at test time; no client content.
"""

import os
import types

from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import app.utils.document_loader as dl
from app.utils.document_loader import SafePyPDFLoader, get_loader

_PDF_MIME = "application/pdf"

# Running header/footer text -- IDENTICAL on every page, so the pass sees them
# repeat and drops them. (A page-number footer that varies per page is a LIMIT; see
# the module report.)
_HEADER = "RUNNING_HEADER_confidential"
_FOOTER = "RUNNING_FOOTER_boilerplate"

# Column x-starts: a gap far wider than _PDF_COLUMN_GAP_PT (72) so clustering is
# unambiguous. Header/footer share the left x; they are removed before clustering.
_LEFT_X = 72
_RIGHT_X = 330


def _two_column_page(writer, left_lines, right_lines, header=_HEADER, footer=_FOOTER):
    """A page whose content stream INTERLEAVES the two columns row by row (how a
    two-column page is commonly authored and exactly what pypdf mis-orders), with a
    running header at the top band and a running footer at the bottom band."""
    page = writer.add_blank_page(width=612, height=792)
    ops = ["BT /F1 12 Tf"]
    ops.append("1 0 0 1 %d 760 Tm (%s) Tj" % (_LEFT_X, header))  # top band
    y = 700
    for left, right in zip(left_lines, right_lines):
        ops.append("1 0 0 1 %d %d Tm (%s) Tj" % (_LEFT_X, y, left))
        ops.append("1 0 0 1 %d %d Tm (%s) Tj" % (_RIGHT_X, y, right))
        y -= 20
    ops.append("1 0 0 1 %d 30 Tm (%s) Tj" % (_LEFT_X, footer))   # bottom band
    ops.append("ET")

    stream = DecodedStreamObject()
    stream.set_data(" ".join(ops).encode("latin-1"))
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


def _left(p, n):
    return "P%d_LEFT_%d" % (p, n)


def _right(p, n):
    return "P%d_RIGHT_%d" % (p, n)


def make_two_column_pdf_SYNTHETIC(path, pages=2, rows=3, header=_HEADER, footer=_FOOTER):
    """`pages` two-column pages, each with distinct body tokens (so the body is not
    mistaken for a running line) and the SAME header/footer (so they are)."""
    writer = PdfWriter()
    for p in range(pages):
        left = [_left(p, i) for i in range(rows)]
        right = [_right(p, i) for i in range(rows)]
        _two_column_page(writer, left, right, header=header, footer=footer)
    with open(path, "wb") as fh:
        writer.write(fh)


def _load(path, ocr_budget=None):
    loader, known, ext = get_loader("doc.pdf", _PDF_MIME, str(path))
    assert known is True and ext == "pdf"
    return loader.load()


def _pages_text(docs):
    return {d.metadata.get("page"): d.page_content for d in docs}


# ---------------------------------------------------------------------------
# Precondition: the defect is real -- raw pypdf does NOT read in author order.
# ---------------------------------------------------------------------------


def test_raw_pypdf_native_order_is_not_author_order_SYNTHETIC(tmp_path):
    """Drives the fixture through RAW PyPDFLoader (no E2 pass) to prove the fixture
    reproduces the defect: some right-column text precedes some left-column text.
    This is the baseline the mutation control returns to."""
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=1, rows=3)
    raw = PyPDFLoader(str(path)).load()
    text = "\n".join(d.page_content for d in raw)
    # The author order is L0,L1,L2 then R0,R1,R2. Raw pypdf interleaves, so R0
    # appears before L2 -- i.e. NOT author order.
    assert text.find(_right(0, 0)) < text.find(_left(0, 2)), (
        "fixture did not reproduce the column-interleaving defect: %r" % text
    )


# ---------------------------------------------------------------------------
# 1. The pass yields author (column) reading order.
# ---------------------------------------------------------------------------


def test_two_column_reading_order_is_author_order_SYNTHETIC(tmp_path):
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=2, rows=3)
    by_page = _pages_text(_load(path))

    for p in (0, 1):
        text = by_page[p]
        order = [text.find(tok) for tok in
                 (_left(p, 0), _left(p, 1), _left(p, 2),
                  _right(p, 0), _right(p, 1), _right(p, 2))]
        assert all(i >= 0 for i in order), (p, order, text)
        assert order == sorted(order), (
            "page %d not in column reading order (L0,L1,L2,R0,R1,R2): %r" % (p, text)
        )


# ---------------------------------------------------------------------------
# 2. Running headers/footers are removed -- and the removal is by REPETITION,
#    not by band alone (a once-only band line is KEPT).
# ---------------------------------------------------------------------------


def test_running_header_footer_removed_when_repeated_SYNTHETIC(tmp_path):
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=3, rows=3)  # >= _PDF_HEADER_FOOTER_MIN_REPEAT
    for text in _pages_text(_load(path)).values():
        assert _HEADER not in text, "running header not removed: %r" % text
        assert _FOOTER not in text, "running footer not removed: %r" % text
        # the body survived
        assert "LEFT_0" in text and "RIGHT_2" in text, text


def test_a_once_only_band_line_is_kept_SYNTHETIC(tmp_path):
    """Control for the repetition rule: a header that appears on ONLY ONE page is
    below _PDF_HEADER_FOOTER_MIN_REPEAT and must NOT be dropped -- otherwise the
    pass would eat a genuine one-off title, not a running header."""
    path = tmp_path / "onepage.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=1, rows=3, header="ONE_OFF_TITLE")
    text = _pages_text(_load(path))[0]
    assert "ONE_OFF_TITLE" in text, (
        "a band line appearing on a single page was dropped; repetition rule broken: %r" % text
    )


# ---------------------------------------------------------------------------
# 3. Per-page locators and text_source are unchanged.
# ---------------------------------------------------------------------------


def test_page_locators_and_text_source_unchanged_SYNTHETIC(tmp_path):
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=2, rows=3)
    docs = _load(path)

    # page indices intact and complete.
    assert sorted(d.metadata.get("page") for d in docs) == [0, 1]
    for d in docs:
        # total_pages / page_label are pypdf's, never touched by the pass.
        assert d.metadata.get("total_pages") == 2
        assert "page_label" in d.metadata
        # A native page is never labelled OCR by the reorder pass.
        assert d.metadata.get("text_source") != "ocr"


# ---------------------------------------------------------------------------
# 4. THE OCR BOUNDARY -- proven, not asserted.
# ---------------------------------------------------------------------------


def test_a_scanned_page_reaches_the_layout_pass_empty_SYNTHETIC(tmp_path):
    """The native/OCR divergence exists BEFORE the pass runs: `_apply_layout`
    receives a scanned page as EMPTY text and passes it through unchanged (it never
    reads positions or reorders it). Driven at the method to pin the boundary
    directly."""
    from langchain_core.documents import Document

    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=1, rows=3)
    loader = SafePyPDFLoader(str(path))

    scanned = Document(page_content="", metadata={"page": 0})
    out = list(loader._apply_layout(iter([scanned])))
    assert len(out) == 1
    assert out[0].page_content == "", "the pass altered an empty (scanned) page"


def test_ocr_output_is_not_reordered_by_the_layout_pass_SYNTHETIC(tmp_path, monkeypatch):
    """If the reorder ever applied to OCR output, this reddens. A page with NO native
    text goes to OCR downstream of the pass; the OCR text is deliberately shaped like
    scrambled columns, and it must arrive VERBATIM -- never column-reordered."""
    # A blank page: no text layer, so native text is empty and OCR is reached.
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)

    # OCR text that WOULD be reordered if the column pass touched it (right-ish token
    # first). Its exact string is the assertion oracle.
    ocr_text = "RIGHT_COL_A\nLEFT_COL_A\nRIGHT_COL_B\nLEFT_COL_B"
    fake = types.SimpleNamespace(
        text=ocr_text, reason="ocr", attempted=True, confidence=0.9,
        images_seen=1, notes=[],
    )
    monkeypatch.setattr("app.utils.ocr.ocr_page", lambda page, budget: fake)

    # Construct with a non-None OCR budget so the OCR branch is reached.
    loader = SafePyPDFLoader(str(path), ocr_budget=object())
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata.get("text_source") == "ocr"
    assert docs[0].page_content == ocr_text, (
        "OCR output was altered by the native layout pass -- the boundary leaked: %r"
        % docs[0].page_content
    )


# ---------------------------------------------------------------------------
# 5. Bound: above _PDF_LAYOUT_MAX_PAGES the pass is skipped (pages pass through).
# ---------------------------------------------------------------------------


def test_layout_pass_is_skipped_above_the_page_cap_SYNTHETIC(monkeypatch):
    from langchain_core.documents import Document

    monkeypatch.setattr(dl, "_PDF_LAYOUT_MAX_PAGES", 1)
    # The cap check returns before the file is opened, so no real PDF is needed.
    loader = SafePyPDFLoader("unused.pdf")

    # Two synthetic native docs > cap of 1 -> pass-through, page_content untouched.
    d0 = Document(page_content="B_before\nA_before", metadata={"page": 0})
    d1 = Document(page_content="D_before\nC_before", metadata={"page": 1})
    out = list(loader._apply_layout(iter([d0, d1])))
    assert [d.page_content for d in out] == ["B_before\nA_before", "D_before\nC_before"]


# ---------------------------------------------------------------------------
# 6. Guard the guards -- the reorder and the drop actually do something, so a
#    green above can never be vacuous.
# ---------------------------------------------------------------------------


def test_columns_text_orders_and_drops_positive_control():
    loader = SafePyPDFLoader("unused.pdf")
    # runs as (x, y, text): two columns, interleaved input; a header in the drop set.
    runs = [
        (72, 760, _HEADER),
        (72, 700, "L0"), (330, 700, "R0"),
        (72, 680, "L1"), (330, 680, "R1"),
    ]
    out = loader._columns_text(runs, drop_keys={loader._run_key(_HEADER)})
    assert out == "L0\nL1\nR0\nR1", out  # column order, header dropped

    # ...and with an empty drop set the header would still be present and first.
    out2 = loader._columns_text(runs, drop_keys=set())
    assert out2.splitlines()[0] == _HEADER and "L0" in out2


def test_running_header_footer_keys_needs_repetition():
    loader = SafePyPDFLoader("unused.pdf")
    # Same band line on two pages -> caught; a one-page band line -> not.
    per_page = [
        ([(72, 760, _HEADER), (72, 700, "P0_body")], 792.0),
        ([(72, 760, _HEADER), (72, 700, "P1_body")], 792.0),
        ([(72, 760, "ONE_OFF"), (72, 700, "P2_body")], 792.0),
    ]
    keys = loader._running_header_footer_keys(per_page)
    assert loader._run_key(_HEADER) in keys
    assert loader._run_key("ONE_OFF") not in keys
    assert loader._run_key("P0_body") not in keys  # mid-page body never a header

"""PDF multi-column reading order + running header/footer de-duplication (E2).

WHAT E2 ADDS. pypdf reads a two-column page in row order, gluing the columns
together (right-column text spliced between left-column lines) -- wrong reading
order for retrieval and citation. And a running header/footer repeated on every
page is extracted on every page, inflating the index with boilerplate. E2 adds a
NATIVE-PATH-ONLY layout pass to SafePyPDFLoader that reorders each native page into
column reading order and drops running headers/footers.

POSITION SOURCE (see the PR body / the E2 probe). pypdf's visitor aggregates each
text ROW into a single call reporting only the row's FIRST x, so an interleaved
two-column row arrives GLUED at one x and cannot be separated -- the visitor cannot
do column ordering. `extraction_mode="layout"` CAN: it renders columns with a wide
run of spaces between them, which this pass splits on. Header/footer BANDS use the
visitor's per-row y, which is accurate, so a repeated mid-page line is never
mistaken for a footer.

GROUND TRUTH is the AUTHOR order: left column top-to-bottom, then right column. All
fixtures are hand-built PDFs whose every text run's x/y is set with explicit PDF
operators (`BT/Tf/Tm/Tj/ET`) through pypdf -- NO reportlab, no new dependency.

All fixtures are SYNTHETIC and generated at test time; no client content.
"""

import types

from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import app.utils.document_loader as dl
from app.utils.document_loader import SafePyPDFLoader, get_loader

_PDF_MIME = "application/pdf"

_HEADER = "RUNNING_HEADER_confidential"
_FOOTER = "RUNNING_FOOTER_boilerplate"
_LEFT_X, _RIGHT_X = 72, 330


def _font_resources(writer, page):
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources


def _page(writer, runs):
    """runs = list of (x, y, text). Content stream is written in the given order."""
    page = writer.add_blank_page(width=612, height=792)
    ops = ["BT /F1 12 Tf"]
    for x, y, text in runs:
        ops.append("1 0 0 1 %d %d Tm (%s) Tj" % (x, y, text))
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data(" ".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    _font_resources(writer, page)
    return page


def _left(p, n):
    return "P%d_LEFT_%d" % (p, n)


def _right(p, n):
    return "P%d_RIGHT_%d" % (p, n)


def make_two_column_pdf_SYNTHETIC(path, pages=2, rows=3, header=_HEADER, footer=_FOOTER):
    """`pages` two-column pages, each with distinct body tokens and the SAME
    header/footer. Content stream INTERLEAVES the columns row by row (how a
    two-column page is commonly authored and what pypdf mis-orders)."""
    writer = PdfWriter()
    for p in range(pages):
        runs = [(_LEFT_X, 760, header)]
        y = 700
        for i in range(rows):
            runs.append((_LEFT_X, y, _left(p, i)))
            runs.append((_RIGHT_X, y, _right(p, i)))
            y -= 20
        runs.append((_LEFT_X, 30, footer))
        _page(writer, runs)
    with open(path, "wb") as fh:
        writer.write(fh)


def _load(path):
    loader, known, ext = get_loader("doc.pdf", _PDF_MIME, str(path))
    assert known is True and ext == "pdf"
    return loader.load()


def _pages_text(docs):
    return {d.metadata.get("page"): d.page_content for d in docs}


# ---------------------------------------------------------------------------
# Precondition: the defect is real -- raw pypdf does NOT read in author order.
# ---------------------------------------------------------------------------


def test_raw_pypdf_native_order_is_not_author_order_SYNTHETIC(tmp_path):
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=1, rows=3)
    text = "\n".join(d.page_content for d in PyPDFLoader(str(path)).load())
    assert text.find(_right(0, 0)) < text.find(_left(0, 2)), (
        "fixture did not reproduce the column-interleaving defect: %r" % text
    )


# ---------------------------------------------------------------------------
# 1. The pass yields author (column) reading order, no doubled blank lines.
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
        assert "\n\n" not in text, "doubled blank lines in reordered page: %r" % text


# ---------------------------------------------------------------------------
# 2. Running headers/footers removed by REPETITION (a one-off band line is kept).
# ---------------------------------------------------------------------------


def test_running_header_footer_removed_when_repeated_SYNTHETIC(tmp_path):
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=3, rows=3)
    for text in _pages_text(_load(path)).values():
        assert _HEADER not in text, "running header not removed: %r" % text
        assert _FOOTER not in text, "running footer not removed: %r" % text
        assert "LEFT_0" in text and "RIGHT_2" in text, text


def test_a_once_only_band_line_is_kept_SYNTHETIC(tmp_path):
    """A header appearing on ONLY ONE page is below the repeat threshold and must
    NOT be dropped -- otherwise the pass eats a genuine one-off title."""
    path = tmp_path / "onepage.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=1, rows=3, header="ONE_OFF_TITLE")
    text = _pages_text(_load(path))[0]
    assert "ONE_OFF_TITLE" in text, "a single-page band line was dropped: %r" % text


# ---------------------------------------------------------------------------
# BLOCKER 2 regression: a single-column page with a centered title, a
# right-aligned date and a deep indent must be UNCHANGED (no false multi-column).
# ---------------------------------------------------------------------------


def make_single_column_ornamented_pdf_SYNTHETIC(path):
    writer = PdfWriter()
    _page(writer, [
        (250, 760, "CENTERED_TITLE"),        # centered heading
        (430, 730, "2026-09-23_RIGHT_DATE"), # right-aligned date
        (72, 700, "BODY_LINE_ONE_left"),
        (72, 680, "BODY_LINE_TWO_left"),
        (140, 660, "DEEPLY_INDENTED_line"),  # deep indent
        (72, 640, "BODY_LINE_THREE_left"),
    ])
    with open(path, "wb") as fh:
        writer.write(fh)


def test_single_column_ornamented_page_is_unchanged_SYNTHETIC(tmp_path):
    path = tmp_path / "ornament.pdf"
    make_single_column_ornamented_pdf_SYNTHETIC(str(path))
    raw = "\n".join(d.page_content for d in PyPDFLoader(str(path)).load())
    got = "\n".join(d.page_content for d in _load(path))
    assert got == raw, (
        "single-column page was rewritten (Blocker 2). raw=%r got=%r" % (raw, got)
    )


# ---------------------------------------------------------------------------
# BLOCKER 3 regression: a running-header PHRASE that also appears as genuine
# mid-body text on another page must SURVIVE there (drop only in-band).
# ---------------------------------------------------------------------------


def make_header_phrase_also_in_body_pdf_SYNTHETIC(path):
    writer = PdfWriter()
    shared = "SHARED_PHRASE_alpha"
    # pages 0,1: SHARED is a running header (top band, y=760)
    for p in (0, 1):
        _page(writer, [(72, 760, shared), (72, 700, "P%d_body_x" % p), (72, 680, "P%d_body_y" % p)])
    # page 2: SHARED is genuine MID-BODY text (y=680), not in any band
    _page(writer, [(72, 760, "P2_intro_top"), (72, 680, shared), (72, 400, "P2_outro_low")])
    with open(path, "wb") as fh:
        writer.write(fh)
    return shared


def test_body_line_matching_a_running_header_survives_SYNTHETIC(tmp_path):
    path = tmp_path / "hf_body.pdf"
    shared = make_header_phrase_also_in_body_pdf_SYNTHETIC(str(path))
    by_page = _pages_text(_load(path))
    assert shared not in by_page[0], "running header not removed on page 0: %r" % by_page[0]
    assert shared not in by_page[1], "running header not removed on page 1: %r" % by_page[1]
    assert shared in by_page[2], (
        "Blocker 3: mid-body text matching a running-header phrase was deleted: %r" % by_page[2]
    )


# ---------------------------------------------------------------------------
# 3. Per-page locators and text_source are unchanged.
# ---------------------------------------------------------------------------


def test_page_locators_and_text_source_unchanged_SYNTHETIC(tmp_path):
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=2, rows=3)
    docs = _load(path)
    assert sorted(d.metadata.get("page") for d in docs) == [0, 1]
    for d in docs:
        assert d.metadata.get("total_pages") == 2
        assert "page_label" in d.metadata
        assert d.metadata.get("text_source") != "ocr"


# ---------------------------------------------------------------------------
# BLOCKER 4 / MAJOR: the OCR boundary -- proven structurally, not by construction.
# ---------------------------------------------------------------------------


def test_a_scanned_page_reaches_the_layout_pass_empty_SYNTHETIC(tmp_path):
    from langchain_core.documents import Document

    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=1, rows=3)
    loader = SafePyPDFLoader(str(path))
    scanned = Document(page_content="", metadata={"page": 0})
    out = list(loader._apply_layout(iter([scanned])))
    assert len(out) == 1 and out[0].page_content == ""


def test_ocr_output_is_structurally_unreachable_from_the_layout_pass_SYNTHETIC(tmp_path, monkeypatch):
    """NON-VACUOUS: spies on what `_apply_layout` actually yields. A scanned page is
    OCR'd DOWNSTREAM of the pass, so the OCR text must appear in the final output but
    NEVER in what `_apply_layout` yielded. If the pass were moved onto the OCR path,
    the OCR text WOULD appear in the spy and this reddens."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)  # no text layer -> OCR path
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)

    ocr_text = "RIGHT_COL_A\nLEFT_COL_A\nRIGHT_COL_B\nLEFT_COL_B"
    fake = types.SimpleNamespace(
        text=ocr_text, reason="ocr", attempted=True, confidence=0.9, images_seen=1, notes=[],
    )
    monkeypatch.setattr("app.utils.ocr.ocr_page", lambda page, budget: fake)

    seen_by_pass = []
    orig = SafePyPDFLoader._apply_layout

    def spy(self, pages):
        for d in orig(self, pages):
            seen_by_pass.append(d.page_content)
            yield d

    monkeypatch.setattr(SafePyPDFLoader, "_apply_layout", spy)

    loader = SafePyPDFLoader(str(path), ocr_budget=object())
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata.get("text_source") == "ocr"
    assert docs[0].page_content == ocr_text
    assert all(ocr_text not in seen for seen in seen_by_pass), (
        "OCR text passed THROUGH the layout pass -- boundary leaked: %r" % seen_by_pass
    )


# ---------------------------------------------------------------------------
# 4. Bound: above _PDF_LAYOUT_MAX_PAGES the pass is skipped (raw order kept).
# ---------------------------------------------------------------------------


def test_layout_pass_is_skipped_above_the_page_cap_SYNTHETIC(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_PDF_LAYOUT_MAX_PAGES", 1)
    path = tmp_path / "twocol.pdf"
    make_two_column_pdf_SYNTHETIC(str(path), pages=2, rows=3)  # 2 pages > cap 1
    text = _pages_text(_load(path))[0]
    # Not reordered: right-column token still precedes a later left-column token.
    assert text.find(_right(0, 0)) < text.find(_left(0, 2)), (
        "page was reordered despite exceeding the page cap: %r" % text
    )


# ---------------------------------------------------------------------------
# 5. Guard the guards -- unit-level positive controls for the new helpers.
# ---------------------------------------------------------------------------


def test_reorder_multicolumn_orders_and_needs_two_columns():
    loader = SafePyPDFLoader("unused.pdf")
    layout = (
        "L0" + " " * 40 + "R0\n"
        "L1" + " " * 40 + "R1\n"
        "L2" + " " * 40 + "R2\n"
    )
    out = loader._reorder_multicolumn(layout, drop_keys=set(), band_keys=set())
    assert out == "L0\nL1\nL2\nR0\nR1\nR2", out
    # A single-column layout (no wide gaps) is NOT multi-column -> None.
    assert loader._reorder_multicolumn("just one column line\nand another\n",
                                       set(), set()) is None


def test_reorder_drops_a_band_key_only_in_band():
    loader = SafePyPDFLoader("unused.pdf")
    layout = (
        "HDR\n"
        "L0" + " " * 40 + "R0\n"
        "L1" + " " * 40 + "HDR\n"   # 'HDR' appears mid-body on the last row too
    )
    key = loader._run_key("HDR")
    out = loader._reorder_multicolumn(layout, drop_keys={key}, band_keys={key})
    assert "HDR" in out, out          # the body-cell HDR survived
    assert out.count("HDR") == 1, out  # only the header instance was removed


def test_split_cells_and_cluster_columns():
    loader = SafePyPDFLoader("unused.pdf")
    cells = loader._split_cells("LEFT" + " " * 20 + "RIGHT")
    assert [t for _o, t in cells] == ["LEFT", "RIGHT"]
    assert loader._cluster_columns([o for o, _t in cells])[0] == 0
    assert len(loader._cluster_columns([0, 1, 2])) == 1        # no wide gap -> one column
    assert len(loader._cluster_columns([0, 40])) == 2          # wide gap -> two columns


def test_dedup_single_column_drops_only_in_band_first_last():
    loader = SafePyPDFLoader("unused.pdf")
    key = loader._run_key("HEAD")
    out = loader._dedup_single_column("HEAD\nbody one\nbody two", {key}, {key})
    assert out == "body one\nbody two", out
    # Not in this page's band -> nothing dropped (returns None).
    assert loader._dedup_single_column("HEAD\nbody", {key}, set()) is None

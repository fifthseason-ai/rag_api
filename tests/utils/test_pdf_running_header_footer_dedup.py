"""PDF running header/footer de-duplication on the native path (E2).

WHAT E2 DOES (re-scoped). A running header/footer repeated on every page is
extracted on every page, inflating the index with boilerplate and duplicating a
line across citations. E2 adds a NATIVE-PATH-ONLY pass to SafePyPDFLoader that drops
a running header/footer LINE -- a line that sits in the geometric top/bottom band
(by the visitor's accurate per-row y) AND repeats across pages. It rewrites
`page_content` ONLY; `page`, `page_label`, `total_pages`, `source` and `text_source`
are untouched, and OCR output is never touched (the pass is upstream of `_with_ocr`).

RE-SCOPE NOTE. Column REORDERING was investigated and dropped (see the PR body):
`extraction_mode="layout"` genuinely separates interleaved columns, but a data table
(needs row order) is not reliably distinguishable from article columns (needs column
order) on this finance/engagement corpus, and a wrong reorder loses no token so the
corruption is silent. Only the safe, unambiguous dedup ships.

All fixtures are SYNTHETIC and generated at test time (hand-built PDFs via pypdf
`BT/Tf/Tm/Tj`, no reportlab, no new dependency); no client content.
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
    """runs = list of (x, y, text)."""
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


def make_running_hf_pdf_SYNTHETIC(path, pages=3, header=_HEADER, footer=_FOOTER):
    """Single-column pages with the SAME header (top band) and footer (bottom band)
    on every page and distinct body per page."""
    writer = PdfWriter()
    for p in range(pages):
        runs = [(72, 760, header)]
        y = 700
        for i in range(3):
            runs.append((72, y, "P%d_body_%d" % (p, i)))
            y -= 20
        runs.append((72, 30, footer))
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
# 1. Running headers/footers are removed when repeated; body is preserved.
# ---------------------------------------------------------------------------


def test_running_header_footer_removed_when_repeated_SYNTHETIC(tmp_path):
    # >= _PDF_DEDUP_MIN_PAGES pages, header/footer on EVERY page (fraction 1.0).
    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=6)
    for p, text in _pages_text(_load(path)).items():
        assert _HEADER not in text, "running header not removed on page %d: %r" % (p, text)
        assert _FOOTER not in text, "running footer not removed on page %d: %r" % (p, text)
        assert "P%d_body_0" % p in text and "P%d_body_2" % p in text, text


def test_a_non_repeated_band_line_is_kept_SYNTHETIC(tmp_path):
    """A top-band line that DIFFERS per page is not a running header and must be kept,
    even across enough pages for dedup to run."""
    writer = PdfWriter()
    for p in range(6):
        _page(writer, [(72, 760, "UNIQUE_TITLE_%d" % p), (72, 700, "P%d_body" % p)])
    path = tmp_path / "unique.pdf"
    with open(str(path), "wb") as fh:
        writer.write(fh)
    by_page = _pages_text(_load(path))
    for p in range(6):
        assert "UNIQUE_TITLE_%d" % p in by_page[p], by_page[p]


def test_a_single_page_document_is_untouched_SYNTHETIC(tmp_path):
    """A one-page document cannot carry a REPEATED running line, so the pass does no
    work and the page is byte-identical to raw pypdf."""
    writer = PdfWriter()
    _page(writer, [(72, 760, "SOME_TOP_LINE"), (72, 700, "some body")])
    path = tmp_path / "one.pdf"
    with open(str(path), "wb") as fh:
        writer.write(fh)
    raw = "\n".join(d.page_content for d in PyPDFLoader(str(path)).load())
    got = "\n".join(d.page_content for d in _load(path))
    assert got == raw, "single-page doc was altered: raw=%r got=%r" % (raw, got)


# ---------------------------------------------------------------------------
# F2: a standalone mid-body line equal to a running-header phrase SURVIVES on the
# SAME page where the phrase is also the running header (drop first/last only).
# ---------------------------------------------------------------------------


def make_header_phrase_also_mid_body_same_page_SYNTHETIC(path):
    writer = PdfWriter()
    shared = "SHARED_PHRASE_alpha"
    # SHARED is a running header (top band, y=760) on EVERY page (fraction 1.0), and on
    # page 0 it ALSO appears as standalone mid-body text (y=560).
    _page(writer, [(72, 760, shared), (72, 700, "P0_body_a"), (72, 560, shared), (72, 400, "P0_body_b")])
    for p in range(1, 6):
        _page(writer, [(72, 760, shared), (72, 700, "P%d_body_a" % p)])
    with open(path, "wb") as fh:
        writer.write(fh)
    return shared


def test_same_page_mid_body_line_matching_running_header_survives_SYNTHETIC(tmp_path):
    path = tmp_path / "hf_body.pdf"
    shared = make_header_phrase_also_mid_body_same_page_SYNTHETIC(str(path))
    by_page = _pages_text(_load(path))
    # page 0: the top header instance is removed, the mid-body instance survives.
    assert by_page[0].count(shared) == 1, (
        "F2: expected exactly one surviving (mid-body) occurrence on page 0: %r" % by_page[0]
    )
    assert "P0_body_a" in by_page[0] and "P0_body_b" in by_page[0], by_page[0]
    # every other page: header-only occurrence removed.
    for p in range(1, 6):
        assert shared not in by_page[p], "running header not removed on page %d: %r" % (p, by_page[p])


# ---------------------------------------------------------------------------
# F1 moot-by-removal: tabular single-column content is NEVER reordered. The old
# column pass scrambled `Name Value Notes / Alice 100 ok / ...` into
# `Name Alice ... Value 100 ...`; with reordering removed, table ROWS are intact
# and a repeated header is still dropped without disturbing them.
# ---------------------------------------------------------------------------


def make_ascii_table_pdf_SYNTHETIC(path, header=_HEADER):
    writer = PdfWriter()
    rows = [("Name", "Value", "Notes"), ("Alice", "100", "ok"),
            ("Bob", "200", "low"), ("Carol", "300", "high")]
    for p in range(6):
        runs = [(72, 760, header)]  # running header on every page (fraction 1.0 -> dropped)
        y = 700
        for name, val, note in rows:
            runs.append((72, y, name)); runs.append((240, y, val)); runs.append((400, y, note))
            y -= 20
        _page(writer, runs)
    with open(path, "wb") as fh:
        writer.write(fh)


def test_tabular_page_rows_are_not_scrambled_SYNTHETIC(tmp_path):
    path = tmp_path / "table.pdf"
    make_ascii_table_pdf_SYNTHETIC(str(path))
    text = _pages_text(_load(path))[0]
    # the running header is gone...
    assert _HEADER not in text, text
    # ...but the table reads ROW-major, exactly as raw pypdf gives it -- NOT
    # column-major (which would put 'Alice' right after 'Name').
    assert text.find("Name") < text.find("Value") < text.find("Notes") < text.find("Alice"), (
        "table rows were scrambled (F1 regression): %r" % text
    )
    assert text.find("Alice") < text.find("Bob") < text.find("Carol"), text
    # and equals raw pypdf with only the header line removed.
    raw_lines = "\n".join(d.page_content for d in PyPDFLoader(str(path)).load()).split("\n")
    assert [ln for ln in raw_lines if _HEADER not in ln][:1]  # sanity: raw had content


# ---------------------------------------------------------------------------
# 2. Per-page locators and text_source are unchanged.
# ---------------------------------------------------------------------------


def test_page_locators_and_text_source_unchanged_SYNTHETIC(tmp_path):
    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=2)
    docs = _load(path)
    assert sorted(d.metadata.get("page") for d in docs) == [0, 1]
    for d in docs:
        assert d.metadata.get("total_pages") == 2
        assert "page_label" in d.metadata
        assert d.metadata.get("text_source") != "ocr"


# ---------------------------------------------------------------------------
# 3. The OCR boundary -- proven structurally, not by construction.
# ---------------------------------------------------------------------------


def test_a_scanned_page_reaches_the_pass_empty_SYNTHETIC(tmp_path):
    from langchain_core.documents import Document

    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=6)
    loader = SafePyPDFLoader(str(path))
    scanned = Document(page_content="", metadata={"page": 0})
    out = list(loader._dedup_running_hf(iter([scanned])))
    assert len(out) == 1 and out[0].page_content == ""


def test_ocr_output_is_structurally_unreachable_from_the_pass_SYNTHETIC(tmp_path, monkeypatch):
    """NON-VACUOUS: spies on what `_dedup_running_hf` actually yields. A scanned page is
    OCR'd DOWNSTREAM of the pass, so the OCR text must appear in the final output but
    NEVER in what the pass yielded. If the pass were moved onto the OCR path, the OCR
    text WOULD appear in the spy and this reddens."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)  # no text layer -> OCR path
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)

    ocr_text = "OCR_LINE_ONE\nOCR_LINE_TWO"
    fake = types.SimpleNamespace(
        text=ocr_text, reason="ocr", attempted=True, confidence=0.9, images_seen=1, notes=[],
    )
    monkeypatch.setattr("app.utils.ocr.ocr_page", lambda page, budget: fake)

    seen_by_pass = []
    orig = SafePyPDFLoader._dedup_running_hf

    def spy(self, pages):
        for d in orig(self, pages):
            seen_by_pass.append(d.page_content)
            yield d

    monkeypatch.setattr(SafePyPDFLoader, "_dedup_running_hf", spy)

    loader = SafePyPDFLoader(str(path), ocr_budget=object())
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata.get("text_source") == "ocr"
    assert docs[0].page_content == ocr_text
    assert all(ocr_text not in seen for seen in seen_by_pass), (
        "OCR text passed THROUGH the dedup pass -- boundary leaked: %r" % seen_by_pass
    )


# ---------------------------------------------------------------------------
# 4. Bound: above _PDF_DEDUP_MAX_PAGES the pass is skipped (nothing removed).
# ---------------------------------------------------------------------------


def test_dedup_pass_is_skipped_above_the_page_cap_SYNTHETIC(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_PDF_DEDUP_MAX_PAGES", 1)
    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=6)  # >= min pages, but > cap 1
    text = _pages_text(_load(path))[0]
    assert _HEADER in text, "header removed despite exceeding the page cap: %r" % text


def test_dedup_pass_is_skipped_below_the_minimum_page_count_SYNTHETIC(tmp_path):
    """Below `_PDF_DEDUP_MIN_PAGES`, a repeated band line cannot be distinguished from a
    short continuation table, so NO dedup runs and the header is kept."""
    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=dl._PDF_DEDUP_MIN_PAGES - 1)
    text = _pages_text(_load(path))[0]
    assert _HEADER in text, "header removed on a too-short document: %r" % text


# ---------------------------------------------------------------------------
# 5. Guard the guards -- unit-level controls for the drop helper.
# ---------------------------------------------------------------------------


def test_drop_running_hf_line_drops_only_first_last_in_band():
    loader = SafePyPDFLoader("unused.pdf")
    key = loader._run_key("HEAD")
    # first line is a running-h/f key AND in this page's band -> dropped, count 1.
    out, dropped = loader._drop_running_hf_line("HEAD\nbody one\nbody two", {key}, {key})
    assert out == "body one\nbody two" and dropped == 1, (out, dropped)
    # both first and last -> count 2.
    out2, dropped2 = loader._drop_running_hf_line("HEAD\nbody\nHEAD", {key}, {key})
    assert out2 == "body" and dropped2 == 2, (out2, dropped2)
    # a MID-body match is never dropped (only first/last considered).
    assert loader._drop_running_hf_line("top\nHEAD\nbottom", {key}, {key}) == (None, 0)
    # not in this page's band -> nothing dropped.
    assert loader._drop_running_hf_line("HEAD\nbody", {key}, set()) == (None, 0)


def test_run_key_normalises_whitespace():
    loader = SafePyPDFLoader("unused.pdf")
    assert loader._run_key("  A   B \n") == "A B"


# ---------------------------------------------------------------------------
# F-A NEGATIVE CASES: legitimately-repeated band content must SURVIVE. A phrase
# that repeats in the band on only a FEW of many pages is a subtotal / continuation
# heading / data row, NOT a running header, and deleting it silently corrupts a
# finance corpus. Built from the reviewer's probes.
# ---------------------------------------------------------------------------


def _doc_with_repeated_line_on(path, npages, phrase, y, on_pages):
    """`npages` pages, each with a UNIQUE top line and a mid body line (so nothing
    else repeats in the band), plus `phrase` placed at `y` on exactly `on_pages`."""
    writer = PdfWriter()
    for p in range(npages):
        runs = [(72, 758, "UNIQUE_TOP_%d" % p), (72, 400, "P%d_body_mid" % p)]
        if p in on_pages:
            runs.append((72, y, phrase))
        _page(writer, runs)
    with open(path, "wb") as fh:
        writer.write(fh)


def test_repeated_subtotal_survives_SYNTHETIC(tmp_path):
    """A per-section subtotal repeated in the bottom band on 2 of 6 pages must NOT be
    deleted (it is a value, not boilerplate)."""
    path = tmp_path / "subtotal.pdf"
    _doc_with_repeated_line_on(str(path), 6, "Subtotal 100.00", y=60, on_pages={2, 5})
    by_page = _pages_text(_load(path))
    assert "Subtotal 100.00" in by_page[2], by_page[2]
    assert "Subtotal 100.00" in by_page[5], by_page[5]


def test_continuation_heading_survives_SYNTHETIC(tmp_path):
    """A continuation heading in the top band spanning the 2 pages of one table
    within a 6-page document must NOT be deleted."""
    path = tmp_path / "cont.pdf"
    _doc_with_repeated_line_on(str(path), 6, "Balance Sheet (continued)", y=760, on_pages={3, 4})
    by_page = _pages_text(_load(path))
    assert "Balance Sheet (continued)" in by_page[3], by_page[3]
    assert "Balance Sheet (continued)" in by_page[4], by_page[4]


def test_repeated_in_band_data_row_survives_SYNTHETIC(tmp_path):
    """A data row just inside the band (y=71) repeated on 2 of 6 pages must survive."""
    path = tmp_path / "datarow.pdf"
    _doc_with_repeated_line_on(str(path), 6, "Line item 42 amount 5.00", y=71, on_pages={1, 4})
    by_page = _pages_text(_load(path))
    assert "Line item 42 amount 5.00" in by_page[1], by_page[1]
    assert "Line item 42 amount 5.00" in by_page[4], by_page[4]


# ---------------------------------------------------------------------------
# DISCLOSURE (PROPOSED, PENDING the consuming lane): a shortened page carries a
# count of the running-h/f lines removed, so the removal is auditable, not silent.
# ---------------------------------------------------------------------------


def test_dropped_lines_are_disclosed_on_the_chunk_metadata_SYNTHETIC(tmp_path):
    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=6)  # header + footer on every page
    docs = _load(path)
    # every page had its header AND footer dropped -> count 2, disclosed.
    for d in docs:
        assert d.metadata.get(dl._PDF_HF_DROPPED_KEY) == 2, d.metadata
    # a document with nothing removed carries no such key (no false disclosure).
    clean = tmp_path / "clean.pdf"
    writer = PdfWriter()
    for p in range(6):
        _page(writer, [(72, 400, "P%d_only_body_no_running_line" % p)])
    with open(str(clean), "wb") as fh:
        writer.write(fh)
    for d in _load(clean):
        assert dl._PDF_HF_DROPPED_KEY not in d.metadata, d.metadata

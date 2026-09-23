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
    path = tmp_path / "hf.pdf"
    make_running_hf_pdf_SYNTHETIC(str(path), pages=3)
    for p, text in _pages_text(_load(path)).items():
        assert _HEADER not in text, "running header not removed on page %d: %r" % (p, text)
        assert _FOOTER not in text, "running footer not removed on page %d: %r" % (p, text)
        assert "P%d_body_0" % p in text and "P%d_body_2" % p in text, text


def test_a_non_repeated_band_line_is_kept_SYNTHETIC(tmp_path):
    """A top-band line that DIFFERS per page is not a running header and must be kept."""
    writer = PdfWriter()
    for p in (0, 1):
        _page(writer, [(72, 760, "UNIQUE_TITLE_%d" % p), (72, 700, "P%d_body" % p)])
    path = tmp_path / "unique.pdf"
    with open(str(path), "wb") as fh:
        writer.write(fh)
    by_page = _pages_text(_load(path))
    assert "UNIQUE_TITLE_0" in by_page[0], by_page[0]
    assert "UNIQUE_TITLE_1" in by_page[1], by_page[1]


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
    # page 0: SHARED as running header (top band, y=760) AND as mid-body text (y=560)
    _page(writer, [(72, 760, shared), (72, 700, "P0_body_a"), (72, 560, shared), (72, 400, "P0_body_b")])
    # page 1: SHARED only as running header
    _page(writer, [(72, 760, shared), (72, 700, "P1_body_a")])
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
    # page 1: header-only occurrence removed.
    assert shared not in by_page[1], "running header not removed on page 1: %r" % by_page[1]


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
    for p in (0, 1):
        runs = [(72, 760, header)]  # running header (repeats -> dropped)
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
    make_running_hf_pdf_SYNTHETIC(str(path), pages=2)
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
    make_running_hf_pdf_SYNTHETIC(str(path), pages=2)  # 2 pages > cap 1
    text = _pages_text(_load(path))[0]
    assert _HEADER in text, "header removed despite exceeding the page cap: %r" % text


# ---------------------------------------------------------------------------
# 5. Guard the guards -- unit-level controls for the drop helper.
# ---------------------------------------------------------------------------


def test_drop_running_hf_line_drops_only_first_last_in_band():
    loader = SafePyPDFLoader("unused.pdf")
    key = loader._run_key("HEAD")
    # first line is a running-h/f key AND in this page's band -> dropped.
    out = loader._drop_running_hf_line("HEAD\nbody one\nbody two", {key}, {key})
    assert out == "body one\nbody two", out
    # a MID-body match is never dropped (only first/last considered).
    mid = loader._drop_running_hf_line("top\nHEAD\nbottom", {key}, {key})
    assert mid is None, mid  # HEAD is neither first nor last -> nothing changes
    # not in this page's band -> nothing dropped.
    assert loader._drop_running_hf_line("HEAD\nbody", {key}, set()) is None


def test_run_key_normalises_whitespace():
    loader = SafePyPDFLoader("unused.pdf")
    assert loader._run_key("  A   B \n") == "A B"

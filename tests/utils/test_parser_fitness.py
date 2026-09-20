"""Real parser-fitness tests for the rag_api /embed ingestion pipeline (KI-02 WP-C).

Goal (Richard, verbatim intent): "Real parser tests for PPTX/PDF/DOCX/XLSX,
including tables, notes, scans, corruption and exact citations. Empty extraction
must not count as success."

Everything here uses SYNTHETIC, deterministic fixtures generated at test time
(python-pptx, openpyxl, pypdf, raw OOXML via zipfile). No client content, no real
documents. The tests prove, per format:

  * extraction completeness (text present) and table / chart / notes semantics,
  * exact citations (PDF `page`, PPTX `slide_number`, XLSX sheet metadata),
  * honest failure on corrupt / encrypted / empty input (never empty success),
  * that `process_documents` page markers match the per-Document `page` metadata,
  * and the route-level empty-extraction guard (zero docs / whitespace ⇒ 4xx,
    NO vector rows written).

Coverage notes recorded in WPC-REPORT.md:
  * XLSX cases were originally gated on the optional `msoffcrypto` package being
    importable, so they skipped in CI while every .xlsx failed in production.
    msoffcrypto-tool is now a pinned requirement and the gate is gone (KI-02
    SP-01.5); tests/utils/test_xlsx_capability.py carries the full Excel suite.
  * OCR of image-only PDFs is env-gated (PDF_EXTRACT_IMAGES=True + rapidocr) and
    not exercised here; the default (False) product behaviour — scan pages yield
    empty text and are caught by the empty-extraction guard — is asserted.
"""

import base64
import io
import os
import zipfile

import pytest
from langchain_core.documents import Document

from app.utils.document_loader import (
    get_loader,
    process_documents,
    SlidePowerPointLoader,
    SafePyPDFLoader,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------------------
# Synthetic fixture generators
# ---------------------------------------------------------------------------


def make_rich_pptx(path):
    """A slide whose evidence sits in FOUR places the goal names: a normal shape
    (title), a table cell, a chart label, and speaker notes."""
    from pptx import Presentation
    from pptx.util import Inches
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title-only layout
    slide.shapes.title.text = "Quarterly Review"

    table = slide.shapes.add_table(
        2, 2, Inches(1), Inches(2), Inches(4), Inches(1)
    ).table
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "EMEA"
    table.cell(1, 1).text = "EUR 4.2M"

    chart_data = CategoryChartData()
    chart_data.categories = ["Alpha", "Beta"]
    chart_data.add_series("Series ONE", (1.0, 2.0))
    slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(1), Inches(4), Inches(4), Inches(3),
        chart_data,
    )

    slide.notes_slide.notes_text_frame.text = "Confidential: EMEA outperformed."
    prs.save(path)


def make_multi_slide_pptx(path):
    """3 slides: two with content, one blank — blank must be skipped, and the
    surviving Documents must be numbered 1 and 3 (citation must track the real
    slide index, not the emitted-document index)."""
    from pptx import Presentation

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Overview"
    s1.placeholders[1].text = "Revenue grew 12% YoY."
    prs.slides.add_slide(prs.slide_layouts[6])  # blank -> skipped
    s3 = prs.slides.add_slide(prs.slide_layouts[1])
    s3.shapes.title.text = "Outlook"
    s3.placeholders[1].text = "Expand into LATAM next year."
    prs.save(path)


def make_empty_pptx(path):
    """A deck with only blank slides — extraction yields ZERO Documents."""
    from pptx import Presentation

    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    prs.slides.add_slide(prs.slide_layouts[6])
    prs.save(path)


# A minimal valid 1x1 PNG. python-pptx reads the IHDR to size the picture; it
# does NOT need Pillow to embed an image, so no runtime dep is added.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _add_image_only_slide(prs):
    """Append a BLANK-layout slide whose only shape is a picture (no text)."""
    from pptx.util import Inches

    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank: no placeholders
    slide.shapes.add_picture(
        io.BytesIO(_PNG_1X1), Inches(1), Inches(1), Inches(2), Inches(2)
    )
    return slide


def make_image_only_pptx(path):
    """A single slide bearing only a picture (no extractable text)."""
    from pptx import Presentation

    prs = Presentation()
    _add_image_only_slide(prs)
    prs.save(path)


def make_mixed_image_text_pptx(path):
    """Slide 1 = picture only (image-only); slide 2 = real text. Both slides must
    remain citable — the image-only slide must not silently disappear."""
    from pptx import Presentation

    prs = Presentation()
    _add_image_only_slide(prs)
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Revenue"
    s2.placeholders[1].text = "EMEA grew 12%."
    prs.save(path)


def make_all_image_pptx(path):
    """Every slide is image-only — extraction yields only empty-but-identifiable
    Documents, which must still trip the per-deck empty-extraction guard."""
    from pptx import Presentation

    prs = Presentation()
    _add_image_only_slide(prs)
    _add_image_only_slide(prs)
    prs.save(path)


def make_native_and_scan_pdf(path):
    """Page 0 = image-only/scanned (no text content stream); page 1 = native
    text. Proves native-vs-scan is distinguishable and page citations are exact."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # page 0: scan-like (no text)
    writer.add_blank_page(width=200, height=200)  # page 1: native text
    page = writer.pages[1]
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
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Native page two revenue 4.2M) Tj ET")
    page[NameObject("/Contents")] = stream
    with open(path, "wb") as f:
        writer.write(f)


def make_all_scan_pdf(path):
    """A PDF whose every page is image-only (no extractable native text)."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


def make_docx(path):
    """DOCX with a heading, body, a 2x2 table, and a header + footer."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    document = (
        f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>'
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Executive Summary</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Body paragraph one.</w:t></w:r></w:p>"
        "<w:tbl>"
        "<w:tr><w:tc><w:p><w:r><w:t>Metric</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>Churn</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>3.1 percent</w:t></w:r></w:p></w:tc></w:tr>"
        "</w:tbl>"
        '<w:sectPr><w:headerReference w:type="default" r:id="rIdH" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
        '<w:footerReference w:type="default" r:id="rIdF" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></w:sectPr>'
        "</w:body></w:document>"
    )
    header = f'<?xml version="1.0"?><w:hdr xmlns:w="{W_NS}"><w:p><w:r><w:t>HEADER confidential</w:t></w:r></w:p></w:hdr>'
    footer = f'<?xml version="1.0"?><w:ftr xmlns:w="{W_NS}"><w:p><w:r><w:t>FOOTER page one</w:t></w:r></w:p></w:ftr>'
    drels = (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdH" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>'
        '<Relationship Id="rIdF" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", drels)
        z.writestr("word/header1.xml", header)
        z.writestr("word/footer1.xml", footer)


def make_multisheet_xlsx(path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws["A1"] = "Item"
    ws["B1"] = "Qty"
    ws["A2"] = "Widget"
    ws["B2"] = 7
    ws2 = wb.create_sheet("Costs")
    ws2["A1"] = "Line"
    ws2["A2"] = "Rent"
    ws2["B2"] = 1200
    wb.save(path)


def make_corrupt_ooxml(path):
    """Bytes that begin with the ZIP magic but are not a valid OOXML package."""
    with open(path, "wb") as f:
        f.write(b"PK\x03\x04 this is not a real office package \x00\x01\x02\x03")


def make_encrypted_like_ooxml(path):
    """An OLE compound-file header — what an encrypted OOXML looks like to
    python-pptx (which cannot open it without msoffcrypto)."""
    with open(path, "wb") as f:
        f.write(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 128)


# ===========================================================================
# PPTX — evidence in normal shapes, a table, a chart label, and speaker notes
# ===========================================================================


def test_pptx_extracts_table_chart_notes_and_title(tmp_path):
    """The four evidence locations the goal names must all be extracted from a
    single slide. Table cells and chart labels are the RED→GREEN fix (WP-C):
    before the fix only the title and notes were extracted."""
    path = tmp_path / "rich.pptx"
    make_rich_pptx(str(path))

    docs = list(SlidePowerPointLoader(str(path)).lazy_load())
    assert len(docs) == 1
    content = docs[0].page_content

    # Normal shape (title)
    assert "Quarterly Review" in content
    # Table cell semantics (header + data cell)
    assert "Region" in content and "Revenue" in content
    assert "EMEA" in content and "EUR 4.2M" in content
    # Chart labels: series name and category labels
    assert "Series ONE" in content
    assert "Alpha" in content and "Beta" in content
    # Speaker notes (product rule: notes ARE included, tagged [Notes])
    assert "[Notes]" in content
    assert "Confidential: EMEA outperformed." in content


def test_pptx_notes_are_included_as_product_rule(tmp_path):
    """Explicit product-rule assertion: presenter notes are INCLUDED in slide
    content (tagged '[Notes] '), not excluded. If this ever flips to a rule that
    excludes notes, this test names the decision that must change."""
    path = tmp_path / "rich.pptx"
    make_rich_pptx(str(path))
    docs = list(SlidePowerPointLoader(str(path)).lazy_load())
    assert any(d.page_content.startswith("Quarterly Review") for d in docs)
    assert any("[Notes] Confidential" in d.page_content for d in docs)


def test_pptx_slide_number_citation_is_exact_and_skips_blank(tmp_path):
    """Every emitted Document carries a slide_number equal to the REAL slide
    index; the blank middle slide is dropped but must not renumber slide 3."""
    path = tmp_path / "deck.pptx"
    make_multi_slide_pptx(str(path))
    docs = list(SlidePowerPointLoader(str(path)).lazy_load())

    assert [d.metadata["slide_number"] for d in docs] == [1, 3]
    assert all("slide_title" in d.metadata for d in docs)
    assert docs[0].metadata["slide_title"] == "Overview"
    assert "LATAM" in docs[1].page_content
    assert docs[1].metadata["slide_number"] == 3


def test_pptx_grouped_shapes_are_walked_recursively():
    """Grouped shapes must be walked so evidence inside a group is not dropped.

    python-pptx cannot AUTHOR a group, so this is a unit test of the recursion
    branch using shapes that mimic the python-pptx API (shape_type / has_table /
    has_chart / has_text_frame / text / shapes). It proves _collect_shape_texts
    descends into a GROUP and collects a nested text frame + a nested table."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    class FakeShape:
        def __init__(self, *, text=None, table=None, chart=None, group=None):
            self._text = text
            self._table = table
            self._chart = chart
            self._group = group
            if group is not None:
                self.shape_type = MSO_SHAPE_TYPE.GROUP
                self.shapes = group
            else:
                self.shape_type = MSO_SHAPE_TYPE.TEXT_BOX
            self.has_text_frame = text is not None
            self.has_table = table is not None
            self.has_chart = chart is not None

        @property
        def text(self):
            return self._text

        @property
        def table(self):
            return self._table

        @property
        def chart(self):
            return self._chart

    class FakeCell:
        def __init__(self, text):
            self.text = text

    class FakeRow:
        def __init__(self, texts):
            self.cells = [FakeCell(t) for t in texts]

    class FakeTable:
        def __init__(self, rows):
            self.rows = [FakeRow(r) for r in rows]

    nested = [
        FakeShape(text="grouped evidence marker"),
        FakeShape(table=FakeTable([["K", "V"], ["nested-cell", "42"]])),
    ]
    top = [
        FakeShape(text="top level text"),
        FakeShape(group=nested),
    ]

    collected = "\n".join(SlidePowerPointLoader._collect_shape_texts(top))
    assert "top level text" in collected
    assert "grouped evidence marker" in collected
    assert "nested-cell" in collected  # table inside the group was walked


def test_pptx_image_only_slide_stays_identifiable(tmp_path):
    """MAJOR-1 (WP-C-F, RED→GREEN): an image-only slide (a picture, no text) must
    NOT be silently dropped. It is emitted as an empty-but-identifiable Document
    carrying the correct slide_number and image_only=True — parity with PDF scan
    pages. Before the fix the slide vanished (0 Documents emitted)."""
    path = tmp_path / "imgonly.pptx"
    make_image_only_pptx(str(path))
    docs = list(SlidePowerPointLoader(str(path)).lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content == ""
    assert docs[0].metadata["slide_number"] == 1
    assert docs[0].metadata.get("image_only") is True


def test_pptx_mixed_image_text_deck_cites_both_slides(tmp_path):
    """MAJOR-1 (RED→GREEN): a mixed deck (image-only slide 1 + text slide 2) must
    cite BOTH slides. Before the fix slide 1 disappeared, only slide_number [2]
    was emitted, and the embed still succeeded — an undisclosed silent drop."""
    path = tmp_path / "mixed.pptx"
    make_mixed_image_text_pptx(str(path))
    docs = list(SlidePowerPointLoader(str(path)).lazy_load())
    assert [d.metadata["slide_number"] for d in docs] == [1, 2]
    # Slide 1: image-only, empty-but-identifiable, marked.
    assert docs[0].page_content == ""
    assert docs[0].metadata.get("image_only") is True
    # Slide 2: real text, NOT marked image_only.
    assert "EMEA grew 12%." in docs[1].page_content
    assert docs[1].metadata.get("image_only") is not True


def test_pptx_truly_blank_slide_is_still_dropped(tmp_path):
    """Boundary regression for MAJOR-1: a slide with NO shapes at all (a truly
    blank spacer) must STILL be dropped and must not renumber the deck, so the
    image-only change never turns blank spacers into phantom empty citations."""
    path = tmp_path / "deck.pptx"
    make_multi_slide_pptx(str(path))  # slide 2 = blank layout-6 slide (no shapes)
    docs = list(SlidePowerPointLoader(str(path)).lazy_load())
    assert [d.metadata["slide_number"] for d in docs] == [1, 3]
    assert all(d.metadata.get("image_only") is not True for d in docs)


# ===========================================================================
# PDF — native text vs scanned/image-only page, and exact page citations
# ===========================================================================


def test_pdf_distinguishes_native_from_scan_and_cites_page(tmp_path):
    """Native page yields text with its exact page number; the scanned (image-
    only) page is identifiable because native text is absent, and it still
    carries its page metadata. Product behaviour with PDF_EXTRACT_IMAGES=False
    (the default): scan pages are NOT OCR'd, so they are empty-but-identifiable
    rather than silently merged into a neighbour."""
    path = tmp_path / "mixed.pdf"
    make_native_and_scan_pdf(str(path))
    docs = SafePyPDFLoader(str(path), extract_images=False).load()

    assert len(docs) == 2
    by_page = {d.metadata["page"]: d for d in docs}
    assert set(by_page) == {0, 1}

    # Scanned page: native text absent (ocr_required territory), but identifiable.
    assert by_page[0].page_content.strip() == ""
    # Native page: text present, cited to the correct page.
    assert "Native page two revenue 4.2M" in by_page[1].page_content


def test_pdf_page_markers_match_page_metadata(tmp_path):
    """process_documents() must emit a '# PAGE n' marker that matches the `page`
    metadata carried on each Document (citation consistency at the shared
    producer). Uses a two-native-page PDF so both markers appear."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    path = tmp_path / "twopage.pdf"
    writer = PdfWriter()
    for i in range(2):
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
        stream.set_data(f"BT /F1 12 Tf 20 100 Td (Text on page {i}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    with open(path, "wb") as f:
        writer.write(f)

    docs = SafePyPDFLoader(str(path), extract_images=False).load()
    # PyPDFLoader is 0-indexed on `page`; process_documents renders that value
    # into "# PAGE n" markers. MINOR-3 fix (WP-C-F): the marker guard now uses
    # `current_page is not None` (was a falsy `current_page` check that dropped
    # the 0-indexed FIRST page's marker), so page 0 gets a "# PAGE 0" marker that
    # matches its metadata, while non-paged formats (page == None) still get none.
    marked = process_documents(docs)
    pages = {d.metadata["page"] for d in docs}
    assert 0 in pages and 1 in pages
    # Both page markers present and matching metadata (page 0 no longer dropped).
    assert "# PAGE 0" in marked
    assert "# PAGE 1" in marked
    assert "Text on page 0" in marked
    assert "Text on page 1" in marked


# ===========================================================================
# DOCX — headings, body, table cells, headers/footers (docx2txt)
# ===========================================================================


def test_docx_extracts_heading_body_table_and_header_footer(tmp_path):
    path = tmp_path / "report.docx"
    make_docx(str(path))
    loader, known_type, ext = get_loader(
        "report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        str(path),
    )
    assert known_type is True and ext == "docx"
    text = " ".join(d.page_content for d in loader.load())

    assert "Executive Summary" in text  # heading
    assert "Body paragraph one." in text  # body
    assert "Metric" in text and "Churn" in text and "3.1 percent" in text  # table
    # Product rule (docx2txt): headers and footers ARE extracted.
    assert "HEADER confidential" in text
    assert "FOOTER page one" in text


# ===========================================================================
# XLSX — routing + sheet citations + honest failure
# (deeper Excel coverage: tests/utils/test_xlsx_capability.py)
# ===========================================================================


def test_xlsx_routes_to_excel_loader(tmp_path):
    """Routing is deterministic and needs no optional deps."""
    from app.utils.document_loader import SheetExcelLoader

    path = tmp_path / "book.xlsx"
    make_multisheet_xlsx(str(path))
    loader, known_type, ext = get_loader(
        "book.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        str(path),
    )
    assert isinstance(loader, SheetExcelLoader)
    assert known_type is True and ext == "xlsx"


def test_xlsx_documents_carry_sheet_citation(tmp_path):
    """get_loader's mode='elements' surfaces the exact
    sheet citation (page_name = sheet name, page_number = sheet index) on every
    Document. Under the default 'single' mode this metadata is entirely absent —
    this asserts the WP-C fix that adds sheet citations."""
    path = tmp_path / "book.xlsx"
    make_multisheet_xlsx(str(path))
    loader, _, _ = get_loader(
        "book.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        str(path),
    )
    docs = loader.load()
    assert len(docs) > 1  # elements mode -> one Document per element, not one blob
    sheet_names = {d.metadata.get("page_name") for d in docs}
    assert {"Sales", "Costs"}.issubset(sheet_names)
    # Every Document carries a sheet citation (name + index).
    for d in docs:
        assert d.metadata.get("page_name") in {"Sales", "Costs"}
        assert isinstance(d.metadata.get("page_number"), int)
    # Cell values reach the right sheet.
    sales_text = " ".join(
        d.page_content for d in docs if d.metadata.get("page_name") == "Sales"
    )
    assert "Widget" in sales_text


# ===========================================================================
# Honest failure — corrupt / encrypted must raise, never yield empty success
# ===========================================================================


def test_corrupt_pptx_raises_not_empty(tmp_path):
    path = tmp_path / "corrupt.pptx"
    make_corrupt_ooxml(str(path))
    with pytest.raises(Exception) as exc:
        list(SlidePowerPointLoader(str(path)).lazy_load())
    # Must be a real parse error, not a silent empty result.
    assert exc.value is not None


def test_encrypted_like_pptx_raises_not_empty(tmp_path):
    path = tmp_path / "enc.pptx"
    make_encrypted_like_ooxml(str(path))
    with pytest.raises(Exception):
        list(SlidePowerPointLoader(str(path)).lazy_load())


# ===========================================================================
# Route-level empty-extraction guard (KI-02 WP-C requirement 3)
#
# "Empty extraction must not count as success": zero documents or zero
# non-empty chunks ⇒ HTTP 4xx with a clear reason and NO vector rows written.
#
# The vector store is SIMULATED here: AsyncPgVector.aadd_documents is monkey-
# patched to a recorder so we can assert (a) the HTTP status and (b) that NO
# insert was attempted. Real embedding/DB is out of scope for a unit run and is
# not needed to prove the guard. RED→GREEN: before the guard, a whitespace-only
# or empty-deck upload returned HTTP 200 {"status": true} with zero rows (empty
# success); the guard makes it a 422 with nothing written.
# ===========================================================================

import datetime  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import jwt  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from app.routes import document_routes  # noqa: E402
from app.routes.document_routes import _assert_extractable_content  # noqa: E402
from app.services.vector_store.async_pg_vector import AsyncPgVector  # noqa: E402

_GUARD_SECRET = "testsecret"


# ---------------------------------------------------------------------------
# MAJOR-2 (WP-C-F): the empty-extraction guard must measure non-emptiness on the
# SAME normalization the pipeline persists (`clean_text`, which strips NUL and
# invalid UTF-8). `str.strip()` alone leaves NUL / lone surrogates intact, so a
# page that is only NUL / invalid-UTF8 would pass the guard and then be cleaned
# to '' and embedded as an empty chunk — "empty extraction = success", the exact
# invariant this increment exists to enforce. These probes are the reviewer's.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", ["\x00\x00\x00", "\udc80\udc80", " \x00 "])
def test_guard_rejects_content_that_cleans_to_empty(probe):
    """RED→GREEN: NUL-only / invalid-UTF8-only / cleans-to-whitespace content
    must raise 422. clean_text('\\x00\\x00\\x00') == '' and
    clean_text('\\udc80\\udc80') == ''. Before the fix the guard used raw
    str.strip() (NUL and surrogates survive strip) and these PASSED the guard."""
    with pytest.raises(HTTPException) as exc:
        _assert_extractable_content([Document(page_content=probe)], "junk.pdf")
    assert exc.value.status_code == 422


def test_guard_allows_content_that_survives_clean_text():
    """Control: text that survives clean_text still passes the guard (no false
    positive) — real content and a mixed doc list with one non-empty member."""
    _assert_extractable_content([Document(page_content="Real revenue 4.2M")], "ok.pdf")
    _assert_extractable_content(
        [
            Document(page_content="\x00\x00\x00"),
            Document(page_content="Genuine text here"),
        ],
        "ok.pdf",
    )


def _guard_hdr(ent, act, tid="tenantA", uid="testuser"):
    os.environ["JWT_SECRET"] = _GUARD_SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ent,
        "act": act,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {
        "Authorization": f"Bearer {jwt.encode(payload, _GUARD_SECRET, algorithm='HS256')}"
    }


@pytest.fixture()
def guard_client(monkeypatch):
    """TestClient with the vector store SIMULATED and every insert recorded."""
    os.environ["JWT_SECRET"] = _GUARD_SECRET
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
    h = _guard_hdr(ent=["userA"], act=["write"])
    return client.post(
        "/embed",
        data={"file_id": "f-guard", "entity_id": "userA"},
        files={"file": (filename, io.BytesIO(content), content_type)},
        headers=h,
    )


def test_embed_whitespace_only_file_rejected_no_rows(guard_client):
    """Whitespace-only text extracts to no non-empty content -> 422, no insert."""
    r = _embed(guard_client, "blank.txt", b"   \n\t  \r\n   ", "text/plain")
    assert r.status_code == 422, r.text
    # WP-G1: the 422 detail is now a superset dict {message, extraction}; the
    # original human message is preserved verbatim under "message".
    assert "No extractable text" in r.json()["detail"]["message"]
    assert guard_client.inserted_batches == []  # NO vector rows written


def test_embed_empty_pptx_rejected_no_rows(guard_client, tmp_path):
    """A deck of only blank slides yields ZERO Documents -> 422, no insert."""
    path = tmp_path / "empty.pptx"
    make_empty_pptx(str(path))
    r = _embed(
        guard_client,
        "empty.pptx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert r.status_code == 422, r.text
    assert guard_client.inserted_batches == []


def test_embed_all_scanned_pdf_rejected_no_rows(guard_client, tmp_path):
    """An all-image/scanned PDF (no native text, no OCR by default) must not be
    a silent empty success -> 422, no insert."""
    path = tmp_path / "scan.pdf"
    make_all_scan_pdf(str(path))
    r = _embed(guard_client, "scan.pdf", path.read_bytes(), "application/pdf")
    assert r.status_code == 422, r.text
    assert guard_client.inserted_batches == []


def test_embed_all_image_pptx_rejected_no_rows(guard_client, tmp_path):
    """MAJOR-1 x guard (WP-C-F): a deck where EVERY slide is image-only now emits
    empty-but-identifiable Documents (not zero); the empty-extraction guard must
    STILL reject it 422 with no rows, so the image-only Documents never become a
    phantom empty success. Keeps the per-deck empty guard honest."""
    path = tmp_path / "allimg.pptx"
    make_all_image_pptx(str(path))
    r = _embed(
        guard_client,
        "allimg.pptx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert r.status_code == 422, r.text
    assert guard_client.inserted_batches == []


def test_embed_nul_only_content_rejected_no_rows(guard_client, monkeypatch):
    """MAJOR-2 route proof (RED→GREEN): content that cleans to empty (NUL /
    invalid-UTF8) must be rejected 422 with NO vector rows. The LOADER is
    SIMULATED here (a real file whose extracted text layer is exclusively NUL is
    impractical to author); the guard, the clean_content=pdf path, and the
    no-write behavior are all real. Before the fix this returned 200 with an
    empty-chunk batch recorded (empty success)."""

    class _NulLoader:
        def __init__(self, *args, **kwargs):
            self._temp_filepath = None

        def load(self):
            return [
                Document(
                    page_content="\x00\x00\x00",
                    metadata={"source": "x", "page": 0},
                )
            ]

        def lazy_load(self):
            return iter(self.load())

    monkeypatch.setattr(
        document_routes,
        "get_loader",
        # ** absorbs ocr_budget (FILES-01): this stub cares only that the route
        # gets a loader, not how the real one is configured.
        lambda filename, content_type, filepath, **_: (_NulLoader(), True, "pdf"),
    )
    r = _embed(guard_client, "junk.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")
    assert r.status_code == 422, r.text
    assert guard_client.inserted_batches == []


def test_embed_corrupt_pptx_is_honest_failure_no_rows(guard_client, tmp_path):
    """A corrupt OOXML must fail honestly (4xx), never as empty success, and
    must write no rows."""
    path = tmp_path / "corrupt.pptx"
    make_corrupt_ooxml(str(path))
    r = _embed(
        guard_client,
        "corrupt.pptx",
        path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert 400 <= r.status_code < 500, r.text
    assert guard_client.inserted_batches == []


def test_embed_real_content_still_succeeds_and_writes(guard_client):
    """Control: a file with real text still embeds (200) and DOES write rows.
    Proves the guard rejects only empty extraction, not valid content."""
    r = _embed(
        guard_client,
        "good.txt",
        b"Revenue grew twelve percent year over year across EMEA.",
        "text/plain",
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] is True
    assert len(guard_client.inserted_batches) >= 1
    assert sum(len(b) for b in guard_client.inserted_batches) >= 1

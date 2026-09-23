"""DOCX extraction quality: reading ORDER, table COMPLETENESS, and TEXT BOX handling
(FILES-DEV P06-3-EXTRACTION-QUALITY, SYNTHETIC fixtures only).

WHY. The existing DOCX coverage (tests/utils/test_parser_fitness.py
::test_docx_extracts_heading_body_table_and_header_footer) asserts token PRESENCE
only -- "Executive Summary" in text, "Churn" in text -- never the ORDER those tokens
come out in, and never a TEXT BOX at all. A deck-derived proposal (the B-1 batch is
PPT/DOCX heavy) routinely puts pull-quotes and callouts in text boxes, and Word/
PowerPoint write a text box as an <mc:AlternateContent> element carrying the SAME
runs twice: a DrawingML <wps:txbx> Choice and a VML <w:pict>/<v:textbox> Fallback,
of which a reader is meant to render exactly ONE.

DOCX has no per-unit locator (F-DOCX1: Docx2txtLoader flattens the whole document to
ONE Document whose metadata is exactly {'source': ...}); its "usable source
reference" is honestly `none`. So the only extraction-quality properties left to
prove for DOCX are ORDER and COMPLETENESS, which is what this file pins.

All fixtures are SYNTHETIC and generated at test time. Synthetic proof establishes
loader behaviour on a KNOWN shape; it is NOT proof against a real client original,
which needs Graph/Box consent that is outstanding (A3/A4).
"""

import os
import zipfile

from app.utils.document_loader import get_loader

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

_CT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="'
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)

# A text box in the form Word/PowerPoint actually write: mc:AlternateContent with a
# DrawingML wps:txbx Choice AND a VML v:textbox Fallback, both carrying the SAME run.
_TEXTBOX = (
    '<w:p><w:r>'
    '<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
    '<mc:Choice Requires="wps" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
    '<w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
    '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
    '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
    '<wps:wsp><wps:txbx><w:txbxContent>'
    '<w:p><w:r><w:t>ECHO_TEXTBOX pull quote</w:t></w:r></w:p>'
    '</w:txbxContent></wps:txbx></wps:wsp>'
    '</a:graphicData></a:graphic></wp:inline></w:drawing>'
    '</mc:Choice>'
    '<mc:Fallback>'
    '<w:pict xmlns:v="urn:schemas-microsoft-com:vml">'
    '<v:shape><v:textbox><w:txbxContent>'
    '<w:p><w:r><w:t>ECHO_TEXTBOX pull quote</w:t></w:r></w:p>'
    '</w:txbxContent></v:textbox></v:shape>'
    '</w:pict>'
    '</mc:Fallback>'
    '</mc:AlternateContent>'
    '</w:r></w:p>'
)


def _write_docx(path, body_inner):
    document = (
        f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>'
        f'{body_inner}'
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", document)


def make_ordered_docx(path):
    """Ground-truth reading order: heading, body-before, 2x2 table (row-major),
    body-after. No text box -- this fixture is only about order + table cells."""
    body = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>ALPHA_HEADING</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>BRAVO_BODY_BEFORE</w:t></w:r></w:p>'
        '<w:tbl>'
        '<w:tr><w:tc><w:p><w:r><w:t>TCELL_A</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>TCELL_B</w:t></w:r></w:p></w:tc></w:tr>'
        '<w:tr><w:tc><w:p><w:r><w:t>TCELL_C</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>TCELL_D</w:t></w:r></w:p></w:tc></w:tr>'
        '</w:tbl>'
        '<w:p><w:r><w:t>DELTA_BODY_AFTER</w:t></w:r></w:p>'
    )
    _write_docx(path, body)


def make_textbox_docx(path):
    """Body paragraph, then a table, then a text box (mc:AlternateContent)."""
    body = (
        '<w:p><w:r><w:t>BRAVO_BODY_BEFORE</w:t></w:r></w:p>'
        '<w:tbl>'
        '<w:tr><w:tc><w:p><w:r><w:t>TCELL_A</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>TCELL_B</w:t></w:r></w:p></w:tc></w:tr>'
        '</w:tbl>'
        f'{_TEXTBOX}'
    )
    _write_docx(path, body)


def _load_text(path):
    loader, known, ext = get_loader(os.path.basename(path), DOCX_MIME, str(path))
    assert known is True and ext == "docx"
    docs = loader.load()
    assert docs, "fixture produced no Documents"
    return "\n".join(d.page_content for d in docs)


def _ordered_text(tmp_path):
    p = tmp_path / "ordered.docx"
    make_ordered_docx(str(p))
    return _load_text(p)


def _textbox_text(tmp_path):
    p = tmp_path / "tb.docx"
    make_textbox_docx(str(p))
    return _load_text(p)


def test_docx_reading_order_is_preserved(tmp_path):
    """Heading before body before table before the paragraph after the table.

    docx2txt walks word/document.xml in document order, so this is EXPECTED to
    hold on current main. It is pinned because nothing asserted DOCX order before,
    and a loader swap could scramble it silently.
    """
    text = _ordered_text(tmp_path)
    idx = {t: text.find(t) for t in
           ("ALPHA_HEADING", "BRAVO_BODY_BEFORE", "TCELL_A", "DELTA_BODY_AFTER")}
    for t, i in idx.items():
        assert i >= 0, f"{t} missing from extracted text: {text!r}"
    assert idx["ALPHA_HEADING"] < idx["BRAVO_BODY_BEFORE"] < idx["TCELL_A"] < idx["DELTA_BODY_AFTER"], idx


def test_docx_table_cells_are_all_present_row_major(tmp_path):
    """Every table cell is extracted, in row-major order, none silently dropped."""
    text = _ordered_text(tmp_path)
    cells = ["TCELL_A", "TCELL_B", "TCELL_C", "TCELL_D"]
    idxs = [text.find(c) for c in cells]
    assert all(i >= 0 for i in idxs), dict(zip(cells, idxs))
    assert idxs == sorted(idxs), f"cells not in row-major order: {dict(zip(cells, idxs))}"


# -- The measured defect: a real Office text box is extracted TWICE ------------
# MEASURED on the merged tree 5816e133 through get_loader -> Docx2txtLoader with
# make_textbox_docx: docx2txt 0.9 walks word/document.xml with ElementTree.iter(),
# which visits every descendant regardless of the markup-compatibility rules. An
# mc:AlternateContent text box carries the SAME runs in both its mc:Choice
# (DrawingML wps:txbx) and its mc:Fallback (VML v:textbox), so the text box's
# content came out TWICE:
#     '...TCELL_B\n\n\n\nECHO_TEXTBOX pull quote\n\nECHO_TEXTBOX pull quote'
# A duplicated text box is indexed twice: retrieval double-counts it and a citation
# reader sees it twice. The fix keeps exactly ONE text-bearing branch per
# AlternateContent before docx2txt sees the file (SafeDocxLoader).


def test_docx_textbox_content_is_extracted(tmp_path):
    """COMPLETENESS: a text box's content must not be silently dropped."""
    text = _textbox_text(tmp_path)
    assert "ECHO_TEXTBOX pull quote" in text, text


def test_docx_textbox_content_is_not_duplicated(tmp_path):
    """FIDELITY: an mc:AlternateContent text box carries the same runs in its
    Choice and its Fallback; exactly one copy must be extracted, never both."""
    text = _textbox_text(tmp_path)
    assert text.count("ECHO_TEXTBOX pull quote") == 1, (
        f"text box duplicated ({text.count('ECHO_TEXTBOX pull quote')}x): {text!r}"
    )


def test_docx_textbox_comes_after_the_table_in_reading_order(tmp_path):
    """The text box sits after the table in the document, and must extract there."""
    text = _textbox_text(tmp_path)
    assert text.find("TCELL_B") < text.find("ECHO_TEXTBOX"), text

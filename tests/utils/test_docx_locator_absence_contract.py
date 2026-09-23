"""DOCX locator ABSENCE contract (E1, addendum 2026-09-23, SYNTHETIC fixtures).

A DOCX per-unit locator (`block_index`) can be ABSENT for two different facts, and
this file MEASURES exactly which of them the /embed receipt can and cannot
distinguish on the wire -- so the CONTRACT DELTA statement is backed by executable
proof rather than an assumption:

  (a) ABSENT BY NATURE -- a header/footer unit has no position in the body reading
      sequence, so its chunk carries NO block_index key at all (absent, not None).
      This IS distinguishable: the document's locator_kind is "block" and the body
      chunks carry contiguous 0-based indices, so a chunk with no block_index in a
      locator_kind="block" document is a header/footer unit, not lost content.

  (b) ABSENT BECAUSE THE FAIL-SAFE FIRED -- the structured walk hit a body block it
      could not represent and degraded to the flat path; the whole document reports
      locator_kind="none" and no chunk carries block_index.

MEASURED COLLISION (stated, not resolved here): a document that legitimately has NO
body blocks (only header/footer, or an empty body) ALSO degrades to
locator_kind="none". So at the document level, locator_kind="none" does NOT by
itself separate a fail-safe degradation from a document that simply had no body
blocks. Fully distinguishing them would require a new receipt signal (a dedicated
`degraded`/structured-walk flag). Adding a field to the receipt is a FILES producer
decision, with Core as the consumer that would read it; it is OUT OF E1 SCOPE and no
such key is added here. The FILES addendum records this limitation.

SYNTHETIC: every fixture is hand-built OOXML generated at test time.
"""

import zipfile

import pytest

from app.routes.document_routes import _UNIT_LOCATOR_KEYS, _extraction_receipt
from app.utils.document_loader import SafeDocxLoader, get_loader

# Reuse existing synthetic builders/helpers rather than inventing new ones.
from tests.utils.test_docx_reading_order import DOCX_MIME, W_NS
from tests.utils.test_docx_sdt_content_control import _customxml, _load, _para
from tests.utils.test_parser_fitness import make_docx  # body + header + footer

_KEY = SafeDocxLoader._DOCX_LOCATOR_KEY
_KIND = SafeDocxLoader._DOCX_LOCATOR_KIND


def make_header_footer_only_docx_SYNTHETIC(path):
    """A DOCX with NO body-level text block -- only a `w:sectPr` referencing a
    header and footer, plus the header/footer parts. docx2txt still reads the
    header/footer text (it scans header*/footer* parts), while the structured walk
    finds no body block to cite. Mirrors make_docx's packaging with an empty body."""
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


# ---------------------------------------------------------------------------
# Point 2 -- header/footer units carry NO block_index. RED-FIRST vs head 57fc800.
# ---------------------------------------------------------------------------


def test_header_footer_units_carry_no_block_index_SYNTHETIC(tmp_path):
    """RED-FIRST for the header/footer ruling. make_docx = heading + body paragraph
    + 2x2 table (6 body blocks) + a header + a footer.

    PRE-CHANGE (head 57fc800): `_structured_units` appended header/footer into the
    unit list and `load()` enumerated ALL of them, so the header/footer chunks
    carried a block_index (0..7). This test FAILS there: there is no chunk WITHOUT
    the key, so the `aux` list is empty. POST-CHANGE: only the 6 body blocks are
    enumerated (0..5 contiguous) and the header/footer text is emitted as unnamed
    unit(s) with NO block_index key."""
    path = tmp_path / "report.docx"
    make_docx(str(path))
    loader, known, ext = get_loader("report.docx", DOCX_MIME, str(path))
    assert known is True and ext == "docx"
    docs = loader.load()

    body = [d for d in docs if _KEY in (d.metadata or {})]
    aux = [d for d in docs if _KEY not in (d.metadata or {})]

    # Body blocks: 0-based contiguous in document order.
    assert [d.metadata[_KEY] for d in body] == list(range(len(body)))
    assert len(body) > 1

    # Header/footer present as unnamed unit(s): text kept, block_index ABSENT.
    assert aux, "header/footer text must be emitted as unnamed unit(s), not dropped"
    aux_text = "\n".join(d.page_content for d in aux)
    assert "HEADER confidential" in aux_text and "FOOTER page one" in aux_text
    for d in aux:
        assert _KEY not in (d.metadata or {}), (
            f"header/footer chunk must carry NO {_KEY!r} key (absent, not None): {d.metadata}"
        )

    # Receipt view: locator_kind is the family; header/footer fold into the single
    # unnamed (None) unit, so units_total = body blocks + 1.
    receipt = _extraction_receipt(docs)
    assert receipt["locator_kind"] == _KIND
    assert receipt["units_total"] == len(body) + 1


# ---------------------------------------------------------------------------
# The MEASURED absence-distinction contract: what the receipt CAN and CANNOT tell.
# ---------------------------------------------------------------------------


def test_receipt_distinguishes_absent_by_nature_but_not_failsafe_vs_no_body_SYNTHETIC(tmp_path):
    """MEASURED locator-absence contract (addendum 2026-09-23). Three documents,
    read at the loader->receipt boundary:

    (1) body + header/footer  -> locator_kind 'block'; body chunks carry a 0-based
        contiguous block_index; the header/footer chunk carries NO block_index.
        => absent-BY-NATURE is distinguishable inside a 'block' document.
    (2) fail-safe degradation (an unrepresentable w:customXml body child) ->
        locator_kind 'none'.
    (3) header/footer ONLY, no body block -> locator_kind 'none'.

    => COLLISION: (2) and (3) report the SAME document-level locator_kind 'none', so
    the receipt does NOT distinguish a fail-safe degradation from a document that
    simply had no body blocks. This pins the limitation the CONTRACT DELTA states;
    separating them needs a new receipt signal (a FILES producer decision, Core the
    consumer), out of E1 scope."""
    # (1) body + header/footer
    p1 = tmp_path / "body_hf.docx"
    make_docx(str(p1))
    l1, _k1, _e1 = get_loader("body_hf.docx", DOCX_MIME, str(p1))
    docs1 = l1.load()
    r1 = _extraction_receipt(docs1)
    keyed = [d for d in docs1 if _KEY in (d.metadata or {})]
    unkeyed = [d for d in docs1 if _KEY not in (d.metadata or {})]
    assert r1["locator_kind"] == _KIND
    assert [d.metadata[_KEY] for d in keyed] == list(range(len(keyed))) and len(keyed) > 1
    assert unkeyed, "the absent-by-nature case needs a header/footer chunk present"

    # (2) fail-safe degradation
    body2 = _para("H1") + _customxml(_para("LOSTX")) + _para("AFT")
    _p2, l2 = _load(tmp_path, "failsafe.docx", body2)
    docs2 = l2.load()
    r2 = _extraction_receipt(docs2)
    assert r2["locator_kind"] == "none"
    # text preserved by the flat path, no block_index anywhere
    joined2 = "\n".join(d.page_content for d in docs2)
    assert "LOSTX" in joined2
    assert all(_KEY not in (d.metadata or {}) for d in docs2)

    # (3) header/footer only, no body block
    p3 = tmp_path / "hf_only.docx"
    make_header_footer_only_docx_SYNTHETIC(str(p3))
    l3, _k3, _e3 = get_loader("hf_only.docx", DOCX_MIME, str(p3))
    docs3 = l3.load()
    r3 = _extraction_receipt(docs3)
    assert r3["locator_kind"] == "none"
    joined3 = "\n".join(d.page_content for d in docs3)
    assert "HEADER confidential" in joined3 or "FOOTER page one" in joined3

    # The distinction that IS on the wire...
    assert r1["locator_kind"] == _KIND != "none"
    # ...and the COLLISION that is NOT: fail-safe and no-body are indistinguishable.
    assert r2["locator_kind"] == r3["locator_kind"] == "none", (
        "fail-safe and no-body documents must be measured as reporting the same "
        "locator_kind 'none' -- if this ever differs, the CONTRACT DELTA limitation "
        "changed and the addendum wording needs rereading"
    )


def make_body_plus_malformed_header_docx_SYNTHETIC(path):
    """A DOCX with a WELL-FORMED body (so the structured walk finds body blocks) and
    a header1.xml that is NOT well-formed XML (an unclosed element). docx2txt reads
    the same header part and raises a ParseError, so the honest flat verdict is a
    hard failure -- not a silent `complete` over the dropped header."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
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
        "<w:p><w:r><w:t>BODYTEXT</w:t></w:r></w:p>"
        '<w:sectPr><w:headerReference w:type="default" r:id="rIdH" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></w:sectPr>'
        "</w:body></w:document>"
    )
    # Malformed: the <w:hdr> element is never closed -> ET.ParseError on parse.
    bad_header = f'<?xml version="1.0"?><w:hdr xmlns:w="{W_NS}"><w:p><w:r><w:t>HDR_SECRET</w:t></w:r></w:p>'
    drels = (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdH" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", drels)
        z.writestr("word/header1.xml", bad_header)


def test_malformed_header_part_degrades_to_flat_not_fake_complete_SYNTHETIC(tmp_path):
    """RED-FIRST for the FOURTH skip-site the loader sweep found: the header/footer
    `part_root is None` branch in `_structured_units` did a bare `continue`, so an
    UNPARSEABLE header part was skipped and the body units were returned.

    PRE-FIX (head f551890): `_structured_units` returned `(body_units, [])` and
    load() reported a `block`/complete document -- turning docx2txt's honest
    ParseError on that same header part into a fake success. This test FAILS there
    (structured is not None; load() does not raise). POST-FIX: `_structured_units`
    returns None and load() degrades to the flat path, where docx2txt raises its
    real verdict (an honest failure, never a silent complete over a dropped part).
    Well-formed header/footer parts never hit this path."""
    path = tmp_path / "bad_hdr.docx"
    make_body_plus_malformed_header_docx_SYNTHETIC(str(path))
    loader, known, ext = get_loader("bad_hdr.docx", DOCX_MIME, str(path))
    assert known is True and ext == "docx"

    # The structured walk does not silently complete over the unparseable header.
    assert loader._structured_units() is None

    # The honest verdict surfaces on the flat path (docx2txt raises ParseError),
    # rather than a `complete` receipt that dropped the header's text.
    with pytest.raises(Exception):
        loader.load()

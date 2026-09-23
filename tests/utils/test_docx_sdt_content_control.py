"""DOCX block-level content controls (w:sdt) and the structured-walk fail-safe
(E1, SYNTHETIC fixtures only).

WHAT THIS PINS. `SafeDocxLoader._body_units` walked only `w:p` and `w:tbl` DIRECT
children of `w:body`. A block-level content control -- `<w:sdt><w:sdtContent>...
paragraphs/tables...</w:sdtContent></w:sdt>`, a direct body child that is NEITHER
tag -- was silently skipped, so its wrapped text was dropped while the surrounding
paragraphs still produced units. `_structured_units` therefore returned a NON-EMPTY
partial list, `_flat_load` never fired, and the /embed receipt reported
`status: complete` over content it had lost. Content controls are common in real
Word templates and forms, so this was ordinary content loss presented as success.

Two properties are pinned here:
  Part A (fidelity): a block-level `w:sdt` is descended into -- its `w:sdtContent`
  paragraphs and tables are read as BODY units, in document order, contiguous with
  the surrounding blocks (tests 1 and 2).
  Part B (fail-safe invariant): if ANY body child the walk cannot faithfully place
  still carries authored text, the WHOLE structured walk abandons to the flat path
  (`_structured_units` -> None -> `_flat_load`), so the text is kept honestly with
  `locator_kind=none` and NO partial/foreign stamp -- never a lossy partial list
  (test 3). Tests 4 and 5 close the two reviewer findings on the flat/exclusion
  paths.

VALUE/KEY RULED (2026-09-23) `block` / `block_index`; nothing here hardcodes them.
Every assertion reads `SafeDocxLoader._DOCX_LOCATOR_KIND` / `._DOCX_LOCATOR_KEY`, so
the loader constant stays the single source of truth.

SYNTHETIC. Every fixture is hand-built OOXML generated at test time, reusing #85's
`_write_docx` builder. Synthetic proof establishes loader behaviour on a KNOWN
shape; it is NOT proof against a real client original (Graph/Box consent
outstanding, A3/A4).
"""

import pytest

from app.routes.document_routes import (
    _UNIT_LOCATOR_KEYS,
    _extraction_receipt,
    _prepare_documents_sync,
)
from app.utils.document_loader import SafeDocxLoader, get_loader

# Reuse #85's synthetic OOXML builder and MIME rather than inventing new ones.
from tests.utils.test_docx_reading_order import DOCX_MIME, _write_docx

# Read the family from the loader (RULED `block`/`block_index`, 2026-09-23) so the
# loader constant stays the single source and no literal of the value is written
# anywhere in this file.
_KEY = SafeDocxLoader._DOCX_LOCATOR_KEY
_KIND = SafeDocxLoader._DOCX_LOCATOR_KIND


# ---------------------------------------------------------------------------
# Synthetic OOXML helpers (all tags live in the w namespace _write_docx declares).
# ---------------------------------------------------------------------------


def _para(text):
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _sdt(inner_blocks):
    """Wrap block-level XML (paragraphs and/or a table) in a block-level content
    control: <w:sdt><w:sdtPr>..</w:sdtPr><w:sdtContent>..</w:sdtContent></w:sdt>.
    w:sdt/w:sdtPr/w:sdtContent/w:tag are all in the w namespace already declared
    by _write_docx, so no extra xmlns is needed."""
    return (
        '<w:sdt><w:sdtPr><w:tag w:val="SYNTHETIC_CC"/></w:sdtPr>'
        f"<w:sdtContent>{inner_blocks}</w:sdtContent></w:sdt>"
    )


def _customxml(inner_blocks):
    """Wrap block-level XML in a legacy `w:customXml` markup element -- a GENUINE
    OOXML body-level container that can be a direct child of w:body and wraps
    paragraphs/tables. It is NOT one of {w:p, w:tbl, w:sdt}, and the loader
    deliberately does not special-case it, so it is the 'unanticipated
    text-bearing body child' the Part B fail-safe must catch (not a fabricated
    nonsense tag). w:customXml/w:uri/w:element are in the already-declared w
    namespace."""
    return (
        '<w:customXml w:uri="urn:synthetic:meta" w:element="Meta">'
        f"{inner_blocks}</w:customXml>"
    )


def _load(tmp_path, name, body):
    path = tmp_path / name
    _write_docx(str(path), body)
    loader, known, ext = get_loader(name, DOCX_MIME, str(path))
    assert known is True and ext == "docx"
    return str(path), loader


def _which(text, tokens):
    hits = [t for t in tokens if t in text]
    assert len(hits) == 1, f"unit text {text!r} matched {hits!r}, expected exactly one"
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Part A -- sdt wrapping a paragraph. RED-FIRST on the pre-fix loader.
# ---------------------------------------------------------------------------


def test_sdt_wrapping_paragraph_is_a_contiguous_body_unit_SYNTHETIC(tmp_path):
    """RED-FIRST for Part A. Body: H1 -> w:sdt(P_INSIDE) -> P_AFTER.

    PRE-FIX (head aa230fc): the sdt is neither w:p nor w:tbl, so `_body_units`
    skips it and P_INSIDE is DROPPED (units = [H1, P_AFTER]). This test FAILS
    there -- the emitted-order assertion cannot match [H1, INSIDE, AFTER].
    POST-FIX: the sdtContent paragraph is read as a body unit at the same level,
    so authored order is [H1, INSIDE, AFTER] with contiguous 0-based indices."""
    H1, INSIDE, AFTER = "H1_HEADING", "P_INSIDE_SDT", "P_AFTER_SDT"
    authored = [H1, INSIDE, AFTER]
    body = _para(H1) + _sdt(_para(INSIDE)) + _para(AFTER)
    _path, loader = _load(tmp_path, "sdt_para.docx", body)

    docs = loader.load()

    # Emitted in authored reading order, none dropped, none duplicated.
    assert [_which(d.page_content, authored) for d in docs] == authored
    # Contiguous 0-based block indices over all units (the sdt content included).
    assert [d.metadata[_KEY] for d in docs] == list(range(len(authored)))
    # The wrapped text specifically is present as exactly one indexed unit.
    inside = [d for d in docs if INSIDE in d.page_content]
    assert len(inside) == 1 and isinstance(inside[0].metadata.get(_KEY), int)

    # ...and it reaches the prepared/stored chunks (fast, in-process; no pgvector).
    prepared = _prepare_documents_sync(
        docs, "syn-file", "syn-user", True, "ORGANIC", "sdt_para.docx"
    )
    assert any(INSIDE in d.page_content for d in prepared)
    assert all(isinstance((d.metadata or {}).get(_KEY), int) for d in prepared)


# ---------------------------------------------------------------------------
# 2. Part A -- sdt wrapping a table: each cell a unit, row-major, contiguous.
# ---------------------------------------------------------------------------


def test_sdt_wrapping_table_yields_each_cell_row_major_SYNTHETIC(tmp_path):
    C00, C01, C10, C11 = "SC_R0C0", "SC_R0C1", "SC_R1C0", "SC_R1C1"
    authored = [C00, C01, C10, C11]
    table = (
        "<w:tbl>"
        f"<w:tr><w:tc>{_para(C00)}</w:tc><w:tc>{_para(C01)}</w:tc></w:tr>"
        f"<w:tr><w:tc>{_para(C10)}</w:tc><w:tc>{_para(C11)}</w:tc></w:tr>"
        "</w:tbl>"
    )
    _path, loader = _load(tmp_path, "sdt_table.docx", _sdt(table))

    docs = loader.load()

    # Each wrapped cell is one unit, row-major, indices 0..3 contiguous.
    assert [_which(d.page_content, authored) for d in docs] == authored
    assert [d.metadata[_KEY] for d in docs] == [0, 1, 2, 3]


def test_nested_and_empty_sdt_are_handled_SYNTHETIC(tmp_path):
    """Defensive Part A shapes: an sdt nested inside another sdt is recursed into,
    and an sdt with NO sdtContent is skipped cleanly (never an error, never a lost
    unit for a control that wraps nothing)."""
    H1, DEEP, AFTER = "H1_HEADING", "P_DEEP_NESTED", "P_AFTER"
    authored = [H1, DEEP, AFTER]
    # sdt( sdt( P_DEEP ) ) between H1 and P_AFTER, plus an sdt with no sdtContent.
    empty_sdt = "<w:sdt><w:sdtPr/></w:sdt>"
    body = _para(H1) + _sdt(_sdt(_para(DEEP))) + empty_sdt + _para(AFTER)
    _path, loader = _load(tmp_path, "sdt_nested.docx", body)

    docs = loader.load()

    assert [_which(d.page_content, authored) for d in docs] == authored
    assert [d.metadata[_KEY] for d in docs] == list(range(len(authored)))


# ---------------------------------------------------------------------------
# 3. Part B -- fail-safe: an unhandled text-bearing body child forces flat path.
#    RED-FIRST on the pre-fix loader.
# ---------------------------------------------------------------------------


def test_unhandled_text_bearing_body_child_forces_flat_fallback_SYNTHETIC(tmp_path):
    """RED-FIRST for Part B. Body: H1 -> <w:futureBlock> carrying text -> P_AFTER.

    PRE-FIX (head aa230fc): `_body_units` silently skips the unhandled child, so
    `_structured_units` returns a NON-EMPTY partial [H1, P_AFTER] and load() stamps
    block_index -- FAILSAFE text is DROPPED. This test FAILS there on BOTH the
    `_structured_units() is None` assertion and the "no block_index" assertion.
    POST-FIX: the walk raises the private sentinel, `_structured_units` returns None,
    load() takes the flat path: the unhandled child's text is KEPT (docx2txt reads
    every w:t descendant), locator_kind is honestly `none`, and NO per-unit locator
    key -- own or foreign -- rides on any chunk. No lossy partial is ever returned."""
    H1, LOST, AFTER = "H1_HEADING", "FAILSAFE_UNHANDLED_TEXT", "P_AFTER"
    unhandled = f"<w:futureBlock>{_para(LOST)}</w:futureBlock>"
    body = _para(H1) + unhandled + _para(AFTER)
    _path, loader = _load(tmp_path, "failsafe.docx", body)

    # The walk abandons to the flat path rather than emit a partial unit list.
    assert loader._structured_units() is None

    docs = loader.load()
    joined = "\n".join(d.page_content for d in docs)
    # No authored text is lost: all three tokens survive via the flat extraction.
    assert LOST in joined, f"fail-safe dropped the unhandled child's text: {joined!r}"
    assert H1 in joined and AFTER in joined
    # Honest locator: none, and NOT a single per-unit locator key on any chunk.
    assert _extraction_receipt(docs)["locator_kind"] == "none"
    for d in docs:
        meta = d.metadata or {}
        for _kind, key in _UNIT_LOCATOR_KEYS:
            assert key not in meta, (
                f"flat-fallback chunk carries per-unit locator key {key!r}: {meta}"
            )


# ---------------------------------------------------------------------------
# 4. Reviewer Finding 2 -- flat-fallback safety on a corrupt/BadZip docx.
# ---------------------------------------------------------------------------


def test_corrupt_docx_flat_fallback_never_stamps_a_locator_SYNTHETIC(tmp_path):
    """A corrupt (non-zip) .docx must take the flat fallback and never stamp a
    per-unit locator: docx2txt raises its honest terminal verdict on an unreadable
    file (a failure reported as a failure, never a stamped partial), and if any text
    is recovered no chunk carries a locator key. Either way no block_index leaks."""
    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"PK\x03\x04 not a valid docx zip payload")
    loader, known, ext = get_loader("corrupt.docx", DOCX_MIME, str(path))
    assert known is True and ext == "docx"

    # The structured walk cannot open it -> None -> flat fallback.
    assert loader._structured_units() is None

    try:
        docs = loader.load()
    except Exception:
        # docx2txt's honest BadZip verdict: no chunk, so nothing can be stamped.
        return
    for d in docs:
        meta = d.metadata or {}
        for _kind, key in _UNIT_LOCATOR_KEYS:
            assert key not in meta, f"flat-fallback chunk carries {key!r}: {meta}"
    assert _extraction_receipt(docs)["locator_kind"] == "none"


# ---------------------------------------------------------------------------
# 5. Reviewer Finding 3 -- exclusion asserted by PRESENCE, catching a None value.
# ---------------------------------------------------------------------------


def test_prepared_docx_chunk_carries_no_foreign_locator_key_by_presence_SYNTHETIC(tmp_path):
    """A stored DOCX unit must carry no OTHER registered locator key, asserted by
    PRESENCE (`key not in metadata`) rather than `.get(key) is not None`. A foreign
    key stamped with None is invisible to the receipt's detection loop yet still
    rides into the chunk; the presence check catches it, closing that gap."""
    H1, P1, P2 = "H1_HEADING", "P1_BODY", "P2_BODY"
    body = _para(H1) + _para(P1) + _para(P2)
    _path, loader = _load(tmp_path, "clean.docx", body)

    docs = loader.load()
    prepared = _prepare_documents_sync(
        docs, "syn-file", "syn-user", True, "ORGANIC", "clean.docx"
    )
    assert prepared, "no prepared chunks -- the exclusion case cannot be expressed"

    foreign = [key for _kind, key in _UNIT_LOCATOR_KEYS if key != _KEY]
    # Control: the presence check DOES see a None-valued key -- the exact case
    # `.get(k) is not None` would miss. If this were false the assertions below
    # would be vacuous for a None-stamped foreign key.
    sample = foreign[0]
    assert sample in {sample: None}

    for d in prepared:
        meta = d.metadata or {}
        # The DOCX family is present (this really is a structured DOCX unit)...
        assert isinstance(meta.get(_KEY), int), meta
        # ...and NO foreign locator key is present by membership (a None value too).
        present_foreign = [k for k in foreign if k in meta]
        assert not present_foreign, (
            f"stored DOCX chunk carries foreign locator key(s) {present_foreign}: {meta}"
        )


# ---------------------------------------------------------------------------
# 6. INTEGRATION requirements for the sdt/fail-safe fix.
#    (i)  structured path loses no text the flat path captures (same input).
#    (ii) the fail-safe fires on a REALISTIC unanticipated OOXML element.
#    Both are RED-FIRST against the pre-sdt-fix loader (head aa230fc).
# ---------------------------------------------------------------------------


def test_structured_path_loses_no_text_the_flat_path_captures_SYNTHETIC(tmp_path):
    """INTEGRATION: on the SAME input, the structured (block-indexed) path must not
    drop any text the flat (docx2txt) path captures. The fixture puts real content
    INSIDE content controls (an sdt-wrapped paragraph and an sdt-wrapped table), so
    the comparison exercises exactly what the fix added -- not a plain document
    where the two paths agree trivially.

    RED-FIRST (head aa230fc, before sdt handling): the structured path skipped the
    sdt content, so SDT_PARA_TEXT / SDT_CELL_A / SDT_CELL_B were absent from the
    structured output while the flat path captured them -- this test FAILS there.
    POST-FIX: the structured output contains every token the flat output does."""
    tokens = ["H1_TOP", "P_PLAIN", "SDT_PARA_TEXT", "SDT_CELL_A", "SDT_CELL_B", "P_BOTTOM"]
    table = (
        "<w:tbl>"
        f"<w:tr><w:tc>{_para('SDT_CELL_A')}</w:tc><w:tc>{_para('SDT_CELL_B')}</w:tc></w:tr>"
        "</w:tbl>"
    )
    body = (
        _para("H1_TOP") + _para("P_PLAIN")
        + _sdt(_para("SDT_PARA_TEXT")) + _sdt(table)
        + _para("P_BOTTOM")
    )
    _path, loader = _load(tmp_path, "compare.docx", body)

    structured_text = "\n".join(d.page_content for d in loader.load())
    flat_text = "\n".join(d.page_content for d in loader._flat_load())

    # Guard: the flat path really captured the tokens, else the subset is vacuous.
    for t in tokens:
        assert t in flat_text, f"flat path did not capture {t!r}: {flat_text!r}"
    # The structured path loses none of them.
    for t in tokens:
        assert t in structured_text, (
            f"structured path DROPPED {t!r} that the flat path captured: {structured_text!r}"
        )


def test_failsafe_fires_on_unanticipated_real_ooxml_customxml_SYNTHETIC(tmp_path):
    """RED-FIRST for Part B on a REALISTIC element, not a fabricated nonsense tag.

    WHY 'unanticipated': `w:customXml` is a genuine OOXML legacy custom-XML markup
    block that can be a direct child of w:body and wraps paragraphs; the loader
    special-cases only w:p / w:tbl / w:sdt, so w:customXml is outside the handled
    set. A real document could carry one, and its text must not be silently dropped.

    RED-FIRST (head aa230fc): `_body_units` skipped the customXml, so
    `_structured_units` returned a NON-EMPTY partial and the customXml text was lost
    -- this test FAILS there (both the `is None` and the text-preserved assertions).
    POST-FIX: the fail-safe fires, `_structured_units` is None, load() takes the flat
    path, the text is kept, locator_kind is honestly none, no block_index is stamped."""
    H1, LOST, AFTER = "H1_TOP", "CUSTOMXML_BODY_TEXT", "P_AFTER"
    body = _para(H1) + _customxml(_para(LOST)) + _para(AFTER)
    _path, loader = _load(tmp_path, "customxml.docx", body)

    assert loader._structured_units() is None
    docs = loader.load()
    joined = "\n".join(d.page_content for d in docs)
    assert LOST in joined, f"fail-safe dropped the customXml text: {joined!r}"
    assert H1 in joined and AFTER in joined
    assert _extraction_receipt(docs)["locator_kind"] == "none"
    for d in docs:
        meta = d.metadata or {}
        for _kind, key in _UNIT_LOCATOR_KEYS:
            assert key not in meta, (
                f"flat-fallback chunk carries per-unit locator key {key!r}: {meta}"
            )


# ---------------------------------------------------------------------------
# 7. The descent/fail-safe invariant is UNIFORM below the body: at the ROW level
#    (w:tbl children that are not w:tr) and the CELL level (w:tr children that are
#    not w:tc). The row-level content-loss defect (a repeating-section content
#    control wrapping w:tr) is the recurrence these pin. RED-FIRST vs head 671c10e.
# ---------------------------------------------------------------------------


def _row_sdt_table(plain_cell, a, b):
    """A w:tbl with a plain row (one cell = `plain_cell`) followed by a REPEATING-
    SECTION content control: <w:sdt><w:sdtContent><w:tr>..</w:tr></w:sdtContent></w:sdt>
    as a child of w:tbl (a real, common Word feature)."""
    plain_row = "<w:tr><w:tc>" + _para(plain_cell) + "</w:tc></w:tr>"
    wrapped_row = _sdt("<w:tr><w:tc>" + _para(a) + "</w:tc><w:tc>" + _para(b) + "</w:tc></w:tr>")
    return "<w:tbl>" + plain_row + wrapped_row + "</w:tbl>"


def test_row_level_sdt_wrapped_row_cells_are_emitted_in_order_SYNTHETIC(tmp_path):
    """RED-FIRST for the repeating-section (row-level sdt) content-loss defect.

    PRE-FIX (head 671c10e): `_block_units` matched only DIRECT w:tr children of the
    table via `findall`, so an sdt-wrapped row was never visited and its cells were
    DROPPED, while the receipt still reported status complete -- exactly the
    content-loss-as-success class, in a path the body-level fail-safe never reached.
    This test FAILS there. POST-FIX: `_table_units` descends the row-level sdt and
    emits its cells in authored order with contiguous block_index."""
    B0, PLAIN, A, B = "BODY0", "PLAINROW", "SDTROW_A", "SDTROW_B"
    authored = [B0, PLAIN, A, B]
    body = _para(B0) + _row_sdt_table(PLAIN, A, B)
    _path, loader = _load(tmp_path, "row_sdt.docx", body)

    docs = loader.load()
    assert [_which(d.page_content, authored) for d in docs] == authored
    assert [d.metadata[_KEY] for d in docs] == list(range(len(authored)))


@pytest.mark.parametrize("case", ["row_level_sdt", "cell_level_sdt"])
def test_structured_path_loses_no_text_at_every_container_level_SYNTHETIC(tmp_path, case):
    """INTEGRATION no-loss guarantee EXTENDED to the container levels the walk now
    guards below the body: the ROW level (a repeating-section sdt wrapping w:tr) and
    the CELL level (an sdt wrapping w:tc). The original no-loss test covered only
    BODY-level sdt -- which is why the row-level defect slipped through. On the SAME
    input, the structured path must not drop any token the flat path captures.

    RED-FIRST (head 671c10e): for `row_level_sdt` the wrapped row's cells, and for
    `cell_level_sdt` the wrapped cell's text, were absent from the structured output
    while the flat path captured them -- both FAIL there."""
    if case == "row_level_sdt":
        tokens = ["B0", "PLAINCELL", "RSDT_A", "RSDT_B"]
        body = _para("B0") + _row_sdt_table("PLAINCELL", "RSDT_A", "RSDT_B")
    else:  # cell_level_sdt: an sdt wrapping a w:tc inside an ordinary row
        wrapped_cell = _sdt("<w:tc>" + _para("CSDT_TEXT") + "</w:tc>")
        row = "<w:tr><w:tc>" + _para("PLAINCELL") + "</w:tc>" + wrapped_cell + "</w:tr>"
        tokens = ["B0", "PLAINCELL", "CSDT_TEXT"]
        body = _para("B0") + "<w:tbl>" + row + "</w:tbl>"

    _path, loader = _load(tmp_path, f"{case}.docx", body)
    structured_text = "\n".join(d.page_content for d in loader.load())
    flat_text = "\n".join(d.page_content for d in loader._flat_load())

    for t in tokens:
        assert t in flat_text, f"{case}: flat path did not capture {t!r}: {flat_text!r}"
    for t in tokens:
        assert t in structured_text, (
            f"{case}: structured path DROPPED {t!r} the flat path captured: {structured_text!r}"
        )


@pytest.mark.parametrize("case", ["stray_p_under_tbl", "stray_elem_under_tr"])
def test_failsafe_fires_on_unhandled_text_bearing_child_inside_a_table_SYNTHETIC(tmp_path, case):
    """RED-FIRST for the fail-safe INSIDE a table. A text-bearing child that is
    neither handled nor an sdt, at the table level (a stray w:p directly under
    w:tbl) or the row level (a stray element directly under w:tr), must degrade the
    WHOLE document to flat -- text preserved, locator_kind none, NO partial stamp --
    never a silent drop.

    PRE-FIX (head 671c10e): `findall(w:tr)` / `findall(w:tc)` skipped these, so the
    text was dropped and `_structured_units` returned a NON-EMPTY partial -- FAILS
    the `is None` assertion here. POST-FIX: `_table_units`/`_row_units` raise the
    sentinel and the document degrades to flat."""
    if case == "stray_p_under_tbl":
        table = (
            "<w:tbl><w:tr><w:tc>" + _para("CELLTXT") + "</w:tc></w:tr>"
            "<w:p><w:r><w:t>STRAY_IN_TABLE</w:t></w:r></w:p></w:tbl>"
        )
        lost = "STRAY_IN_TABLE"
    else:  # stray_elem_under_tr: a text-bearing element that is not w:tc/w:sdt
        table = (
            "<w:tbl><w:tr><w:tc>" + _para("CELLTXT") + "</w:tc>"
            "<w:futureCell><w:p><w:r><w:t>STRAY_IN_ROW</w:t></w:r></w:p></w:futureCell>"
            "</w:tr></w:tbl>"
        )
        lost = "STRAY_IN_ROW"
    body = _para("BODY0") + table
    _path, loader = _load(tmp_path, f"{case}.docx", body)

    # The whole walk abandons rather than emit a partial that drops the stray text.
    assert loader._structured_units() is None

    docs = loader.load()
    joined = "\n".join(d.page_content for d in docs)
    assert lost in joined, f"{case}: fail-safe dropped the stray text: {joined!r}"
    assert "BODY0" in joined and "CELLTXT" in joined
    assert _extraction_receipt(docs)["locator_kind"] == "none"
    for d in docs:
        meta = d.metadata or {}
        for _kind, key in _UNIT_LOCATOR_KEYS:
            assert key not in meta, (
                f"{case}: flat-fallback chunk carries locator key {key!r}: {meta}"
            )

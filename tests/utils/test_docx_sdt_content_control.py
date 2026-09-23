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

VALUE/KEY PENDING. The locator_kind vocabulary value and the cmetadata key are the
FILES lead's decision; nothing here hardcodes them. Every assertion reads
`SafeDocxLoader._DOCX_LOCATOR_KIND` / `._DOCX_LOCATOR_KEY`.

SYNTHETIC. Every fixture is hand-built OOXML generated at test time, reusing #85's
`_write_docx` builder. Synthetic proof establishes loader behaviour on a KNOWN
shape; it is NOT proof against a real client original (Graph/Box consent
outstanding, A3/A4).
"""

from app.routes.document_routes import (
    _UNIT_LOCATOR_KEYS,
    _extraction_receipt,
    _prepare_documents_sync,
)
from app.utils.document_loader import SafeDocxLoader, get_loader

# Reuse #85's synthetic OOXML builder and MIME rather than inventing new ones.
from tests.utils.test_docx_reading_order import DOCX_MIME, _write_docx

# Read the family from the loader so the PENDING value/key ruling is a one-token
# change and no literal of the value is written anywhere in this file.
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

"""XLSX finer-than-sheet precision (cell-range) + merged-cell resolution + header row.

Card E3, OPTION 3 (FILES-lead ruling PACKET-1-E3-XLSX-PLACEMENT-RULING-20260923T042623Z):
`cell_range`, `header` and `header_row` ship as ADDITIVE, OPTIONAL cmetadata fields and
are DELIBERATELY NOT registered in `_UNIT_LOCATOR_KEYS`. XLSX `locator_kind` stays
`"sheet"` -- a stable TYPE TAG; precision lives in the VALUE (`cell_range`), which Core
reads directly. This suite pins that non-promotion executably.

MEASURED base: origin/main 5816e133. SheetExcelLoader emits one Document per sheet
(pinned by test_xlsx_year_cells.sheet_text `len(docs)==1`); row-per-Document splitting is
therefore impossible without reddening existing tests, so the finer locator is additive
per-sheet metadata: a SHEET-QUALIFIED cell-range naming the occupied extent, one range
per sheet chunk.

Fixtures are SYNTHETIC, built at test time with openpyxl. No client content.

RED-FIRST controls (mutations restored byte-identical; sha256 in the return record):
  * cell_range stamp        -> delete the `_annotate` stamp or the `_sheet_locators`
                               range build  -> the cell_range asserts redden.
  * header rule             -> weaken `_detect_header_row` to "first non-empty row"
                               -> the merged-title-skip asserts redden.
  * merged-cell fill        -> replace `ws.cell(...).value = anchor` with `pass`
                               -> `found` never True, `_resolve_merged_cells` returns
                               None, the fill/no-phantom asserts redden.
  * NON-PROMOTION (cond 4)  -> insert `("cell_range","cell_range")` before
                               `("sheet","page_name")` in `_UNIT_LOCATOR_KEYS`
                               -> `test_locator_kind_stays_sheet_*` redden (locator_kind
                               becomes "cell_range").
  * NO-FABRICATED-EXTENT    -> give an empty sheet a degenerate range in
    (cond 5)                  `_sheet_locators` -> the empty-sheet asserts redden.
"""

import os

import pytest

from langchain_core.documents import Document

from app.utils.document_loader import (
    SheetExcelLoader,
    CELL_RANGE_LOCATOR_KEY,
    CELL_RANGE_LOCATOR_KIND,
    XLSX_HEADER_KEY,
    XLSX_HEADER_ROW_KEY,
)
from app.routes.document_routes import _UNIT_LOCATOR_KEYS, _extraction_receipt

# Reuse the proven in-process route driver + loader helpers from the capability suite.
from tests.utils.test_xlsx_capability import (  # noqa: F401 - pytest fixture `client`
    client,
    _embed,
    load_documents,
)

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ===========================================================================
# SYNTHETIC fixtures
# ===========================================================================


def make_SYNTHETIC_merged_workbook(path):
    """Two sheets. Revenue: merged title A1:C1, header row 2, data rows 3-4, cached
    SUM in B5. Detail: header row 1, category "Hardware" vertically merged A2:A4, a
    yyyy year cell at C2. Ground truth is asserted directly by the tests below."""
    import datetime
    import shutil
    import zipfile
    from openpyxl import Workbook

    wb = Workbook()

    rev = wb.active
    rev.title = "Revenue"
    rev["A1"] = "Quarterly Revenue"
    rev.merge_cells("A1:C1")               # horizontal merged title
    rev["A2"] = "Region"
    rev["B2"] = "Q1"
    rev["C2"] = "Q2"
    rev["A3"] = "EMEA"
    rev["B3"] = 100
    rev["C3"] = 110
    rev["A4"] = "AMER"
    rev["B4"] = 200
    rev["C4"] = 210
    rev["A5"] = "Total"
    rev["B5"] = "=SUM(B3:B4)"              # cached value injected below

    det = wb.create_sheet("Detail")
    det["A1"] = "Category"
    det["B1"] = "Item"
    det["C1"] = "Year"
    det["A2"] = "Hardware"
    det.merge_cells("A2:A4")               # vertical merged category label
    det["B2"] = "Widget"
    det["C2"] = datetime.datetime(2016, 1, 1)
    det["C2"].number_format = "yyyy"       # year-only date cell
    det["B3"] = "Gadget"
    det["C3"] = 2017
    det["B4"] = "Gizmo"
    det["C4"] = 2018

    wb.save(path)

    # Give the Revenue SUM the cached result Excel itself would have stored, so the
    # year-copy pass (written from cached values) keeps the total. Revenue is sheet1.
    tmp = str(path) + ".tmp"
    with zipfile.ZipFile(str(path)) as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                assert b"<f>SUM(B3:B4)</f>" in payload
                payload = payload.replace(
                    b"<f>SUM(B3:B4)</f>", b"<f>SUM(B3:B4)</f><v>300</v>"
                )
            zout.writestr(item, payload)
    shutil.move(tmp, str(path))


def make_SYNTHETIC_populated_plus_empty_workbook(path):
    """One populated sheet and one genuinely empty sheet (no occupied extent)."""
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "Data"
    wb.active["A1"] = "x"
    wb.active["B1"] = "y"
    wb.create_sheet("Blank")  # left empty on purpose
    wb.save(str(path))


# ===========================================================================
# 1. Finer-than-sheet cell-range locator -- DETERMINISTIC (no parser dependency).
# ===========================================================================


def test_SYNTHETIC_sheet_locators_expose_the_cell_range_extent(tmp_path):
    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    locators = SheetExcelLoader(str(path))._sheet_locators()

    assert locators["Revenue"][CELL_RANGE_LOCATOR_KEY] == "Revenue!A1:C5"
    assert locators["Detail"][CELL_RANGE_LOCATOR_KEY] == "Detail!A1:C4"
    # Sheet-qualified so it is unique per unit even when two sheets share A1 notation.
    ranges = {v[CELL_RANGE_LOCATOR_KEY] for v in locators.values()}
    assert len(ranges) == 2


# ===========================================================================
# 2. Header-row detection -- DETERMINISTIC.
# ===========================================================================


def test_SYNTHETIC_header_row_detected_and_skips_merged_title(tmp_path):
    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    locators = SheetExcelLoader(str(path))._sheet_locators()

    # Revenue: the merged title on row 1 must NOT be mistaken for the header.
    assert locators["Revenue"][XLSX_HEADER_ROW_KEY] == 2
    assert locators["Revenue"][XLSX_HEADER_KEY] == ["Region", "Q1", "Q2"]
    # Detail: the header is on row 1.
    assert locators["Detail"][XLSX_HEADER_ROW_KEY] == 1
    assert locators["Detail"][XLSX_HEADER_KEY] == ["Category", "Item", "Year"]


def test_detect_header_row_rule_is_first_multi_distinct_row():
    f = SheetExcelLoader._detect_header_row
    # A merged title (one anchor value) is skipped; the next multi-distinct row wins.
    assert f([(1, ["Title", None, None]), (2, ["Region", "Q1", "Q2"])]) == (
        2, ["Region", "Q1", "Q2"])
    # A filled merged title (all identical) is not a header.
    assert f([(1, ["H", "H", "H"]), (2, ["a", "b"])]) == (2, ["a", "b"])
    # A single-column list has no header.
    assert f([(1, ["only"]), (2, ["one"])]) == (None, None)
    # Empty input.
    assert f([]) == (None, None)


# ===========================================================================
# 3. Merged-cell resolution -- DETERMINISTIC via openpyxl on the produced copy.
# ===========================================================================


def test_SYNTHETIC_merged_cells_are_unmerged_and_filled_no_phantom_empties(tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    copy_path = SheetExcelLoader(str(path))._resolve_merged_cells(
        str(path), str(tmp_path)
    )
    assert copy_path is not None, "a workbook with merged cells must produce a copy"

    wb = load_workbook(copy_path, data_only=True)
    try:
        for ws in wb.worksheets:
            assert list(ws.merged_cells.ranges) == [], (
                "%s still has merged ranges: %s" % (ws.title, ws.merged_cells.ranges))
        rev = wb["Revenue"]
        # Horizontal title: every cell of A1:C1 holds the title -> no phantom empties.
        assert rev["A1"].value == "Quarterly Revenue"
        assert rev["B1"].value == "Quarterly Revenue"
        assert rev["C1"].value == "Quarterly Revenue"
        det = wb["Detail"]
        # Vertical category: the label reaches every continuation row.
        assert det["A2"].value == "Hardware"
        assert det["A3"].value == "Hardware"
        assert det["A4"].value == "Hardware"
    finally:
        wb.close()


def test_resolve_merged_cells_is_a_noop_without_merges(tmp_path):
    """No merges -> parse the source untouched (None), the same contract the year
    pass uses; this is what keeps a plain workbook off the copy path."""
    path = tmp_path / "plain.xlsx"
    make_SYNTHETIC_populated_plus_empty_workbook(str(path))

    assert SheetExcelLoader(str(path))._resolve_merged_cells(
        str(path), str(tmp_path)) is None


def test_resolve_merged_cells_is_never_fatal_on_a_bad_workbook(tmp_path):
    """A diagnostic pass must never cost the content: an unreadable file returns
    None (parse the source as is), it does not raise."""
    path = tmp_path / "not.xlsx"
    path.write_bytes(b"PK\x03\x04 not a real package")
    assert SheetExcelLoader(str(path))._resolve_merged_cells(
        str(path), str(tmp_path)) is None


# ===========================================================================
# 4. NEVER FABRICATE AN EXTENT (condition 5): a sheet with no occupied extent
#    carries NO cell_range -- not a degenerate range, not a neighbouring one.
# ===========================================================================


def test_SYNTHETIC_empty_sheet_has_no_cell_range_at_the_locator_level(tmp_path):
    path = tmp_path / "with-blank.xlsx"
    make_SYNTHETIC_populated_plus_empty_workbook(str(path))

    locators = SheetExcelLoader(str(path))._sheet_locators()

    assert "Blank" not in locators, "an empty sheet must not be given a fabricated extent"
    assert locators["Data"][CELL_RANGE_LOCATOR_KEY] == "Data!A1:B1"


def test_SYNTHETIC_present_cell_range_is_wellformed_and_sheet_qualified(tmp_path):
    """WELL-FORMEDNESS -- not a no-fabrication claim (condition 5 is pinned at the
    locator level by the test above, which is proven to redden under the fabricate
    mutation). Every cell_range PRESENT on an emitted document must be a well-formed
    A1-notation range qualified to that document's OWN sheet. Exercised against a
    two-populated-sheet workbook so a cross-sheet mislabel or a malformed range would
    redden it; `checked >= 2` guarantees the assertion actually runs on both sheets
    rather than passing vacuously."""
    import re

    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    docs = load_documents(path)
    pattern = re.compile(r"^.+![A-Z]+\d+:[A-Z]+\d+$")
    checked = 0
    for d in docs:
        cr = d.metadata.get(CELL_RANGE_LOCATOR_KEY)
        if cr is None:
            continue
        assert pattern.match(cr), "malformed cell_range %r" % cr
        assert cr.startswith(str(d.metadata.get("page_name")) + "!"), (
            "%r is not qualified to its own sheet %r"
            % (cr, d.metadata.get("page_name")))
        checked += 1
    assert checked >= 2, "the fixture must exercise cell_range on both sheets"


# ===========================================================================
# 5. The stamp reaches the LOADER Document, and the merged label reaches content.
#    (Parser-dependent: exercises UnstructuredExcelLoader. CODE-TESTED via image.)
# ===========================================================================


def test_SYNTHETIC_every_sheet_document_carries_cell_range_and_header(tmp_path):
    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    docs = load_documents(path)
    expected_range = {"Revenue": "Revenue!A1:C5", "Detail": "Detail!A1:C4"}
    seen = set()
    for d in docs:
        sheet = d.metadata.get("page_name")
        assert d.metadata.get(CELL_RANGE_LOCATOR_KEY) == expected_range[sheet], (
            "%s chunk carries cell_range %r" % (sheet, d.metadata.get(
                CELL_RANGE_LOCATOR_KEY)))
        seen.add(sheet)
    assert seen == {"Revenue", "Detail"}
    rev_headers = {
        tuple(d.metadata.get(XLSX_HEADER_KEY))
        for d in docs
        if d.metadata.get("page_name") == "Revenue" and d.metadata.get(XLSX_HEADER_KEY)
    }
    assert ("Region", "Q1", "Q2") in rev_headers


def test_SYNTHETIC_merged_category_label_reaches_every_row(tmp_path):
    """The anchor value of a vertical merge is not lost: after resolution the
    category appears once per spanned row rather than only on the first."""
    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    docs = load_documents(path)
    detail = " ".join(
        d.page_content for d in docs if d.metadata.get("page_name") == "Detail"
    )
    assert detail.count("Hardware") >= 3, detail


def make_SYNTHETIC_vertical_merge_no_year_workbook(path):
    """A vertical merge and NOTHING else that would trigger the year-cell copy -- so
    this isolates the merge-fill effect from the _year_only_copy path."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "S1"
    ws["A1"] = "Category"
    ws["B1"] = "Item"
    ws["A2"] = "Hardware"
    ws.merge_cells("A2:A4")
    ws["B2"] = "Widget"
    ws["B3"] = "Gadget"
    ws["B4"] = "Gizmo"
    wb.save(str(path))


def test_SYNTHETIC_merged_label_reaches_every_row_no_year(tmp_path):
    """THE OBSERVABLE DIFFERENCE that keeps _resolve_merged_cells (card #91 review).

    MEASURED: UnstructuredExcelLoader does NOT propagate a merged value across its
    span. On this fixture raw extraction yields the category "Hardware" ONCE and
    splits the continuation rows (Gadget, Gizmo) into separate chunks that have lost
    it entirely; the merge fill puts it on every spanned row. This holds WITHOUT any
    year cell, so it is the fill -- not the _year_only_copy re-save -- that matters.

    RED-FIRST CONTROL: replace the anchor fill in _resolve_merged_cells with `pass`
    (or delete the method + call site) and this reddens with count 1, naming the exact
    difference the transformation exists to produce. Nothing else in the suite depends
    on the fill, so this is its sole permanent guard."""
    path = tmp_path / "vmerge.xlsx"
    make_SYNTHETIC_vertical_merge_no_year_workbook(str(path))

    docs = load_documents(path)
    content = " ".join(d.page_content for d in docs)
    assert content.count("Hardware") >= 3, (
        "the merged category was not propagated to every spanned row: %r" % content)


# ===========================================================================
# 6. THE DELIBERATE NON-PROMOTION (condition 4) + units/empty_locators unchanged
#    (condition 6) -- DETERMINISTIC via _extraction_receipt on synthetic chunks.
#    This is the most important pin: it records executably that we chose NOT to
#    promote cell_range, so locator_kind stays the sheet TYPE TAG while the chunks
#    carry the cell_range precision, and the receipt still groups/names by page_name.
# ===========================================================================


def _xlsx_chunks_like_the_store():
    """Two sheets as the store would hold them: a populated sheet carrying page_name
    AND the additive cell_range, and an empty sheet carrying page_name with NO
    cell_range (condition 5 -- no occupied extent, no fabricated position)."""
    return [
        Document(
            page_content="Region Q1 Q2 EMEA 100 110 AMER 200 210",
            metadata={
                "page_name": "Revenue",
                "page_number": 0,
                CELL_RANGE_LOCATOR_KEY: "Revenue!A1:C5",
                XLSX_HEADER_ROW_KEY: 2,
                XLSX_HEADER_KEY: ["Region", "Q1", "Q2"],
            },
        ),
        Document(
            page_content="",  # empty sheet: no content
            metadata={"page_name": "Empty", "page_number": 1},
        ),
    ]


def test_locator_kind_stays_sheet_while_chunks_carry_cell_range():
    """CONDITION 4 -- the deliberate non-promotion, pinned. cell_range is present on
    the chunk yet locator_kind is the stable `sheet` type tag, NOT `cell_range`.
    RED if anyone registers cell_range before sheet in _UNIT_LOCATOR_KEYS."""
    receipt = _extraction_receipt(_xlsx_chunks_like_the_store())
    assert receipt["locator_kind"] == "sheet"
    # Guard the guard: the chunk really does carry the precision the tag ignores.
    chunks = _xlsx_chunks_like_the_store()
    assert any(c.metadata.get(CELL_RANGE_LOCATOR_KEY) for c in chunks)
    # And cell_range is not (accidentally) a registered family.
    assert (CELL_RANGE_LOCATOR_KIND, CELL_RANGE_LOCATOR_KEY) not in _UNIT_LOCATOR_KEYS
    assert CELL_RANGE_LOCATOR_KEY not in {k for _kind, k in _UNIT_LOCATOR_KEYS}


def test_units_and_empty_locators_stay_keyed_by_page_name():
    """CONDITION 6 -- units stay one-per-sheet and empty_locators names sheets by
    page_name (a sheet name), never by cell_range, despite cell_range being present."""
    receipt = _extraction_receipt(_xlsx_chunks_like_the_store())
    assert receipt["units_total"] == 2
    assert receipt["units_extracted"] == 1
    # The empty sheet is named by its page_name, not by any cell_range value.
    assert receipt["empty_locators"] == ["Empty"]
    assert all("!" not in str(loc) for loc in receipt["empty_locators"]), (
        "empty_locators must name sheets by page_name, not by a cell_range")


def test_SYNTHETIC_route_keeps_sheet_locator_kind_and_units_two(client, tmp_path):
    """CONDITION 4 + 6 end-to-end through /embed (in-process store double): the
    stored chunks carry cell_range yet the receipt advertises locator_kind `sheet`
    with two units. Parser+route dependent (CODE-TESTED via image)."""
    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    r = _embed(client, "merged.xlsx", path.read_bytes(), file_id="f-cellrange")
    assert r.status_code == 200, r.text

    stored = [
        dict(d.metadata or {})
        for batch in client.inserted_batches
        for d in batch
    ]
    assert stored, "the workbook must store chunks"
    assert all(m.get("page_name") for m in stored)
    assert all(m.get(CELL_RANGE_LOCATOR_KEY) for m in stored)

    receipt = r.json()["extraction"]
    assert receipt["locator_kind"] == "sheet"      # deliberate non-promotion
    assert receipt["units_extracted"] == 2


# ===========================================================================
# 7. CONDITION 3 -- the stamp reaches the STORED chunk via a real pgvector
#    round-trip on cmetadata (not loader output). DSN-gated: skips cleanly without
#    RAG_TEST_PG_DSN. Modelled on E1's test_parse_is_not_index real-pg tests.
# ===========================================================================

try:
    from tests.utils.test_parse_is_not_index import (
        needs_pg,
        PG_DSN,
        _real_store,
        _client as _real_client,
        _post as _real_post,
        FID,
    )
    _HAVE_REAL_PG_HARNESS = True
except Exception:  # pragma: no cover - the E1 harness is present on this tree
    _HAVE_REAL_PG_HARNESS = False
    PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
    needs_pg = pytest.mark.skipif(True, reason="E1 real-pg harness unavailable")


@needs_pg
@pytest.mark.skipif(not _HAVE_REAL_PG_HARNESS, reason="E1 real-pg harness unavailable")
def test_real_pgvector_stored_cmetadata_carries_cell_range(monkeypatch, tmp_path):
    """Prove cell_range survives to the stored `cmetadata` jsonb, read back with SQL
    from the real table -- not asserted on the loader's in-memory Document."""
    import psycopg2

    path = tmp_path / "merged.xlsx"
    make_SYNTHETIC_merged_workbook(str(path))

    store = _real_store(monkeypatch, "kc_files_e3_cellrange")
    real_client = _real_client(monkeypatch, store)
    r = _real_post(real_client, "/embed", "merged.xlsx", path.read_bytes(), XLSX_MIME)
    assert r.status_code == 200, r.text

    raw = PG_DSN.replace("postgresql+psycopg2://", "postgresql://")
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "SELECT cmetadata FROM langchain_pg_embedding "
            "WHERE cmetadata->>'file_id' = %s",
            (FID,),
        )
        rows = [row[0] for row in cur.fetchall()]

    assert rows, "no stored chunks for this file_id"
    with_range = [m for m in rows if m.get(CELL_RANGE_LOCATOR_KEY)]
    assert with_range, "no stored chunk carried cell_range in cmetadata"
    # Every stored chunk that names a sheet carries the sheet-qualified extent; and
    # locator_kind is NOT stamped on the chunk (it is a receipt field, not cmetadata).
    for m in with_range:
        assert m.get("page_name")
        assert m[CELL_RANGE_LOCATOR_KEY].startswith(m["page_name"] + "!")
        assert "locator_kind" not in m

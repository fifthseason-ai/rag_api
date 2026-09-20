"""A chunk's locator must be true about ALL of that chunk, not about where it starts.

F1's card says "exact slide/sheet locator" and nothing tested the word. Core then found the
consequence at their end: they were re-deriving the slide from chunk TEXT with a regex that takes
the first `## Slide N` heading, so a chunk spanning a slide boundary was attributed to the slide it
STARTS in rather than the slide it IS. They fixed it by preferring `slide_number` from metadata.

That fix is only correct if this service's metadata is actually exact. Two ways it could not be,
neither of which had any coverage:

    MERGE     one chunk carries text from two units, so whatever single locator it has is a lie
              about part of its own content.
    DROP      a unit longer than CHUNK_SIZE splits into several chunks and only some keep the
              locator. A citation into chunk 2 of a long slide would be unanchored -- and it would
              happen only on LONG units, the ones least likely to be in a fixture.

Measured 2026-09-20 against the real loaders and the real splitter: neither occurs. `page`,
`page_name` and `slide_number` survive a split intact, and `split_documents` never merges
across the Documents a loader emits. These tests hold that.

`row` is DELIBERATELY not in that list. The CSV loader does put a `row` key on the chunk, but
on this base `_UNIT_LOCATOR_KEYS` is only (page, slide_number, page_name) -- `row` is not a
citation locator this code resolves, and #36 is what makes it one. An earlier version of this
docstring listed it anyway, describing a property the tree it ships on does not implement.
Independent review caught it. It is the same mistake as reading a stale local `main` and
calling it current: knowledge true on one branch, asserted about another.

PROVENANCE IS DECIDED FROM THE TEXT. Each unit carries a unique marker, and a chunk's true origin
is read from its content -- never from the metadata, which is the thing on trial. Checking the
metadata against itself would prove nothing.

Both defects were injected to confirm these tests can see them (see the PR body): dropping the
locator after the first chunk reddens the DROP tests on both formats; flattening the units before
splitting reddens MERGE on PPTX and MISLABEL on XLSX.
"""

import pytest
from langchain_core.documents import Document

from app.routes.document_routes import _prepare_documents_sync
from app.utils.document_loader import get_loader

PPTX_CT = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Comfortably over CHUNK_SIZE (1500) so the long unit CANNOT survive as one chunk. The filler
# carries no marker of its own, so it can never make a chunk look like it belongs somewhere.
FILLER = "The delivery programme records each increment and its evidence in sequence. " * 40


def _prepare(docs):
    return _prepare_documents_sync(
        docs,
        "exactness-file",
        "exactness-user",
        False,
        "ORGANIC",
        "units",
        None,
        None,
        "exactness-tenant",
    )


def _load(path, filename, content_type):
    loader, _known, _ext = get_loader(filename, content_type, str(path))
    return list(loader.load())


@pytest.fixture
def pptx_units(tmp_path):
    """Three slides; the middle one is far too long to survive as a single chunk."""
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    blank = prs.slide_layouts[6]
    bodies = [
        ["UNIT-ALPHA-1001 opening scope"],
        ["UNIT-BRAVO-2002 start of the long section", FILLER, "UNIT-BRAVO-2003 end of it"],
        ["UNIT-CHARLIE-3003 closing remarks"],
    ]
    for lines in bodies:
        slide = prs.slides.add_slide(blank)
        box = slide.shapes.add_textbox(
            pptx.util.Inches(0.5), pptx.util.Inches(0.5),
            pptx.util.Inches(9), pptx.util.Inches(6),
        )
        tf = box.text_frame
        tf.text = lines[0]
        for extra in lines[1:]:
            tf.add_paragraph().text = extra
    path = tmp_path / "units.pptx"
    prs.save(str(path))
    markers = {
        "UNIT-ALPHA-1001": 1,
        "UNIT-BRAVO-2002": 2,
        "UNIT-BRAVO-2003": 2,
        "UNIT-CHARLIE-3003": 3,
    }
    return _prepare(_load(path, "units.pptx", PPTX_CT)), markers, "slide_number", 2


@pytest.fixture
def xlsx_units(tmp_path):
    """Two sheets; the second holds a cell far longer than CHUNK_SIZE."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Short Sheet"
    ws1.append(["Note"])
    ws1.append(["UNIT-DELTA-4004 brief"])
    ws2 = wb.create_sheet("Long Sheet")
    ws2.append(["Note"])
    ws2.append(["UNIT-ECHO-5005 start " + FILLER + " UNIT-ECHO-5006 end"])
    path = tmp_path / "units.xlsx"
    wb.save(str(path))
    markers = {
        "UNIT-DELTA-4004": "Short Sheet",
        "UNIT-ECHO-5005": "Long Sheet",
        "UNIT-ECHO-5006": "Long Sheet",
    }
    return _prepare(_load(path, "units.xlsx", XLSX_CT)), markers, "page_name", "Long Sheet"


def _units_in(text, markers):
    return {unit for marker, unit in markers.items() if marker in text}


# ---------------------------------------------------------------------------------------
# PRECONDITIONS. Without these the tests below could pass because the interesting case never
# occurred -- a long unit that did not split makes "every chunk of a split unit keeps its
# locator" vacuously true, and this lane has shipped that mistake before.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["pptx_units", "xlsx_units"])
def test_the_long_unit_really_did_split(fixture, request):
    chunks, markers, key, split_unit = request.getfixturevalue(fixture)
    produced = [c for c in chunks if str(c.metadata.get(key)) == str(split_unit)]
    assert len(produced) > 1, (
        "the long unit produced %d chunk(s), so the split case this file exists for did not "
        "happen and the DROP tests below cannot fail. Lengthen FILLER."
        % len(produced)
    )


@pytest.mark.parametrize("fixture", ["pptx_units", "xlsx_units"])
def test_every_marker_survived_the_pipeline(fixture, request):
    """If a marker is missing, a chunk's origin is undecidable and the analysis is blind."""
    chunks, markers, _key, _split = request.getfixturevalue(fixture)
    blob = "\n".join(c.page_content for c in chunks)
    missing = [m for m in markers if m not in blob]
    assert not missing, "markers lost in extraction, so provenance is undecidable: %s" % missing


# ---------------------------------------------------------------------------------------
# The properties themselves.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["pptx_units", "xlsx_units"])
def test_a_chunk_never_carries_content_from_two_units(fixture, request):
    """MERGE. A chunk holding two slides has a locator that is false about half of itself."""
    chunks, markers, key, _split = request.getfixturevalue(fixture)
    merged = [
        (sorted(str(u) for u in _units_in(c.page_content, markers)),
         c.metadata.get(key), c.page_content[:80])
        for c in chunks
        if len(_units_in(c.page_content, markers)) > 1
    ]
    assert not merged, (
        "a stored chunk carries more than one unit, so its locator is a lie about part of its "
        "own content: %r" % (merged,)
    )


@pytest.mark.parametrize("fixture", ["pptx_units", "xlsx_units"])
def test_every_chunk_carries_a_locator_not_just_the_first(fixture, request):
    """DROP, checked over EVERY chunk.

    Deliberately not restricted to chunks carrying a marker. The first version of this
    measurement analysed 4 of 7 PPTX chunks and reported the locator exact -- the 3 it skipped
    were the filler-only middles of the long unit, which is precisely where a dropped locator
    would be. A chunk with no marker is not evidence of nothing; it is a chunk nobody looked at.
    """
    chunks, _markers, key, _split = request.getfixturevalue(fixture)
    assert chunks, "no chunks were produced at all"
    missing = [c.page_content[:80] for c in chunks if c.metadata.get(key) is None]
    assert not missing, (
        "%d of %d chunks have no %s, so a citation into them cannot be anchored: %r"
        % (len(missing), len(chunks), key, missing)
    )


@pytest.mark.parametrize("fixture", ["pptx_units", "xlsx_units"])
def test_the_locator_agrees_with_the_text_it_is_attached_to(fixture, request):
    """MISLABEL. The locator must name the unit the chunk's own text came from."""
    chunks, markers, key, _split = request.getfixturevalue(fixture)
    wrong = []
    for c in chunks:
        units = _units_in(c.page_content, markers)
        if len(units) == 1:
            expected = str(next(iter(units)))
            if str(c.metadata.get(key)) != expected:
                wrong.append((expected, c.metadata.get(key), c.page_content[:80]))
    assert not wrong, "locator disagrees with the unit the text came from: %r" % (wrong,)


def test_a_locator_the_service_does_not_set_is_carried_through_untouched():
    """The service adds its own fields here; it must not eat the loader's locator doing it.

    Scoped deliberately to the LOCATOR. The neighbouring question -- whether a loader key can
    displace a SERVICE field such as `user_id` -- is #39's, and it is still open on this base:
    asserting it here was tried and fails on `main`, because `main` does not carry that fix.
    Leaving it in would have made this branch red for a defect it does not own and does not
    repair. Recorded rather than deleted, because "the test failed so I removed it" and "the
    test belonged to another change" look identical in a diff.
    """
    doc = Document(page_content="body", metadata={"slide_number": 4, "page_label": "iv"})
    prepared = _prepare([doc])
    assert prepared[0].metadata["slide_number"] == 4
    assert prepared[0].metadata["page_label"] == "iv"
    assert prepared[0].metadata["file_id"] == "exactness-file"


# ---------------------------------------------------------------------------------------
# Added after independent review. Two findings, neither reachable through the current
# pipeline, both closing a real edge of what the suite proves rather than a live defect.
# ---------------------------------------------------------------------------------------


@pytest.fixture
def pdf_units(tmp_path):
    """PDF was argued equivalent and never executed. `SafePyPDFLoader` is OURS.

    The review's reasoning was right -- `SafePyPDFLoader` delegates to a strictly per-page
    loader, so `page` survives a split by the same metadata-copy invariant as the others. But
    it is a third CUSTOM loader with zero coverage of this property, so if it were ever changed
    to aggregate pages, nothing here would notice. "Argued equivalent" and "tested" are not the
    same rung, and this file had been quoting the first as the second.
    """
    pypdf = pytest.importorskip("pypdf")
    from .test_scanned_pdf_ocr import _bytes, _text_page

    writer = pypdf.PdfWriter()
    _text_page(writer, ["UNIT-FOXTROT-6006 opening page"])
    # Page two must exceed CHUNK_SIZE so it splits. Short lines, many of them: a PDF text layer
    # is written per line, so the filler has to be broken up rather than handed over as one run.
    long_page = ["UNIT-GOLF-7007 start of the long page"]
    # 110 lines, not 40. At 40 the long page split into exactly two chunks, each holding one of
    # the markers -- so it produced no MARKERLESS middle, and the set-based assertion below had
    # nothing to bite on for PDF. Measured, not guessed: the wrong-locator-on-markerless mutation
    # reddened PPTX and XLSX and left PDF green. A fixture that cannot manufacture the case makes
    # its test decorative.
    long_page += ["The delivery programme records each increment and its evidence." for _ in range(110)]
    long_page += ["UNIT-GOLF-7008 end of the long page"]
    _text_page(writer, long_page)
    _text_page(writer, ["UNIT-HOTEL-8008 closing page"])

    path = tmp_path / "units.pdf"
    with open(str(path), "wb") as fh:
        fh.write(_bytes(writer))

    markers = {
        "UNIT-FOXTROT-6006": 0,   # `page` is 0-indexed
        "UNIT-GOLF-7007": 1,
        "UNIT-GOLF-7008": 1,
        "UNIT-HOTEL-8008": 2,
    }
    return _prepare(_load(path, "units.pdf", "application/pdf")), markers, "page", 1


@pytest.mark.parametrize(
    "fixture,check",
    [("pdf_units", "split"), ("pdf_units", "merge"), ("pdf_units", "drop"), ("pdf_units", "label")],
)
def test_pdf_locator_is_exact_too(fixture, check, request):
    """The same four properties, executed on the PDF path instead of argued about."""
    chunks, markers, key, split_unit = request.getfixturevalue(fixture)
    if check == "split":
        produced = [c for c in chunks if str(c.metadata.get(key)) == str(split_unit)]
        assert len(produced) > 1, (
            "the long page produced %d chunk(s), so the split case did not happen and the DROP "
            "check below is vacuous for PDF" % len(produced)
        )
    elif check == "merge":
        merged = [c.page_content[:80] for c in chunks
                  if len(_units_in(c.page_content, markers)) > 1]
        assert not merged, "a PDF chunk carries more than one page: %r" % (merged,)
    elif check == "drop":
        missing = [c.page_content[:80] for c in chunks if c.metadata.get(key) is None]
        assert not missing, "%d of %d PDF chunks have no `page`: %r" % (
            len(missing), len(chunks), missing)
    else:
        wrong = []
        for c in chunks:
            units = _units_in(c.page_content, markers)
            if len(units) == 1 and str(c.metadata.get(key)) != str(next(iter(units))):
                wrong.append((next(iter(units)), c.metadata.get(key), c.page_content[:80]))
        assert not wrong, "PDF locator disagrees with the page its text came from: %r" % (wrong,)


@pytest.mark.parametrize("fixture", ["pptx_units", "xlsx_units", "pdf_units"])
def test_no_chunk_carries_a_locator_no_unit_has(fixture, request):
    """Closes the markerless-middle hole the review found.

    The DROP test proves a middle chunk has *a* locator; the MISLABEL test skips it, because a
    chunk with no marker has no text-derived unit to compare against. So a defect that stamped a
    PRESENT BUT WRONG locator onto markerless chunks would pass every other assertion here.

    Not reachable through the current pipeline -- `split_documents` hands every sub-chunk of a
    unit the same metadata dict, so siblings cannot diverge. Asserted anyway, because "cannot
    happen today" is a statement about today's splitter and this file is what would notice.

    The check is set-based: the distinct locator values across ALL chunks must be exactly the
    units the markers say exist. A stamped locator outside that set fails without needing any
    chunk to carry a marker.
    """
    chunks, markers, key, _split = request.getfixturevalue(fixture)
    expected = {str(u) for u in markers.values()}
    seen = {str(c.metadata.get(key)) for c in chunks}
    assert seen == expected, (
        "chunks carry locator values %r but the only units that exist are %r -- a value outside "
        "that set was stamped onto a chunk from somewhere else" % (sorted(seen), sorted(expected))
    )

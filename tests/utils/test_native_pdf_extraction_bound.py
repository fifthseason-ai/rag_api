"""A bounded read of a native PDF: does it STOP, does it KEEP what it read, and does it SAY so?

Three properties, and the third is the one that decides whether the other two are safe to deploy.
A bound that stops the work but lets the document report `complete` would convert a resource
problem into a data problem: Core gates on `status`, so a silently truncated file would be
recorded as fully ingested and the missing pages would never be looked for again.

The default configuration is OFF, so every test that wants the bound has to configure it. That is
deliberate -- a control nobody can see working is indistinguishable from no control -- and it is
why the unconfigured case is asserted here too, rather than assumed from "we changed nothing".
"""

import os

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.utils.document_loader import SafePyPDFLoader
from app.utils.extraction_budget import (
    ATTEMPTED_KEY,
    NOT_INCLUDED_KEY,
    STOPPED_KEY,
    ExtractionBudget,
)

LINES = [
    "FIFTH SEASON CONSULTING",
    "Engagement ledger",
    "Client: Northwind Trading Company",
]


def _text_page(writer, label):
    page = writer.add_blank_page(width=612, height=792)
    ops = ["BT /F1 12 Tf"]
    y = 720
    for line in [label] + LINES:
        ops.append("1 0 0 1 72 %d Tm (%s) Tj" % (y, line))
        y -= 19
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data(" ".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources


@pytest.fixture
def pdf_of_10_pages(tmp_path):
    writer = PdfWriter()
    for i in range(10):
        _text_page(writer, "Page %d marker" % (i + 1))
    path = os.path.join(str(tmp_path), "ledger.pdf")
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


# --- the mechanism is OFF until someone configures it ---------------------------------


def test_unconfigured_budget_reads_the_whole_document(pdf_of_10_pages):
    """The shipped default. Asserted rather than assumed: `enabled` is what makes this a no-op,
    and a bug there would quietly bound every deployment that never asked for a bound."""
    budget = ExtractionBudget(max_pages=0, time_budget_seconds=0)
    assert budget.enabled is False
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())
    assert len(docs) == 10
    assert all(STOPPED_KEY not in (d.metadata or {}) for d in docs)


def test_no_budget_at_all_reads_the_whole_document(pdf_of_10_pages):
    docs = list(SafePyPDFLoader(pdf_of_10_pages).lazy_load())
    assert len(docs) == 10


# --- it actually stops, and keeps what it read ----------------------------------------


def test_page_limit_stops_the_read_and_keeps_the_pages_it_read(pdf_of_10_pages):
    budget = ExtractionBudget(max_pages=3, time_budget_seconds=0)
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())

    assert len(docs) == 3, "the bound must stop the read, not merely flag it"
    assert "Page 1 marker" in docs[0].page_content
    assert "Page 3 marker" in docs[2].page_content, "pages already read are KEPT, not discarded"
    assert budget.stopped_reason == "page_limit"


def test_the_stop_is_recorded_with_what_was_not_opened(pdf_of_10_pages):
    budget = ExtractionBudget(max_pages=4, time_budget_seconds=0)
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())
    marker = docs[-1].metadata

    assert marker[STOPPED_KEY] == "page_limit"
    assert marker[ATTEMPTED_KEY] == 4
    assert marker[NOT_INCLUDED_KEY] == 6, (
        "the pages never opened have no locators of their own -- if this count is wrong or "
        "missing, nothing else in the receipt shows they exist"
    )


def test_time_limit_stops_the_read_and_names_itself(pdf_of_10_pages, monkeypatch):
    """A deadline already in the past: the first page is read (a bound below one page still
    reads one, deliberately), then the clock stops the rest."""
    budget = ExtractionBudget(max_pages=0, time_budget_seconds=0.000001)
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())

    assert 1 <= len(docs) < 10
    assert budget.stopped_reason == "time_limit"
    assert docs[-1].metadata[STOPPED_KEY] == "time_limit"


def test_the_first_bound_to_run_out_is_the_one_reported(pdf_of_10_pages):
    """Both bounds configured, the page limit is the tighter one; the reason must name it rather
    than whichever check happens to run last."""
    budget = ExtractionBudget(max_pages=2, time_budget_seconds=600)
    list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())
    assert budget.stopped_reason == "page_limit"


def test_a_bound_below_one_page_still_reads_one_page(pdf_of_10_pages):
    """A misconfiguration must cost coverage, never honesty. Zero pages would make a bounded
    document indistinguishable from one with no extractable text -- which is the 422 refusal path
    and a different, false statement about the file."""
    budget = ExtractionBudget(max_pages=-5, time_budget_seconds=0)
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())
    assert len(docs) == 1
    assert docs[0].page_content.strip(), "the one page it read must still carry its text"


def test_a_document_inside_its_bounds_is_not_marked_stopped(pdf_of_10_pages):
    """The negative case that keeps the marker meaningful: a bound that marks every document is
    no better than no marker at all."""
    budget = ExtractionBudget(max_pages=50, time_budget_seconds=600)
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())
    assert len(docs) == 10
    assert all(STOPPED_KEY not in (d.metadata or {}) for d in docs)
    assert budget.stopped_reason is None


def test_page_count_that_cannot_be_established_is_reported_as_unknown(pdf_of_10_pages):
    """`pages_not_included` is None, never 0, when the file's own page count cannot be read.
    Zero would say "there was nothing more", which is a claim this service cannot make here.

    The failure is made REAL rather than patched in: the file is unlinked while the already-open
    loader keeps streaming from its handle, so the re-open that counts pages genuinely fails.
    An earlier version of this test monkeypatched `_remaining_page_count` to return None, which
    meant it passed identically with the real code mutated to `return 0` -- it was asserting the
    behaviour of its own stub. Mutating that line now reddens this test.
    """
    budget = ExtractionBudget(max_pages=2, time_budget_seconds=0)
    loader = SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget)

    docs = []
    for index, doc in enumerate(loader.lazy_load()):
        if index == 0:
            os.remove(pdf_of_10_pages)
        docs.append(doc)

    assert budget.stopped_reason == "page_limit"
    assert docs[-1].metadata[NOT_INCLUDED_KEY] is None, (
        "the count could not be established, and unknown is not zero"
    )


# --- what the caller is told ----------------------------------------------------------


def _receipt(docs):
    from app.routes.document_routes import _extraction_receipt

    return _extraction_receipt(docs)


def test_a_stopped_read_is_reported_partial_never_complete(pdf_of_10_pages):
    """THE LOAD-BEARING ONE. Core gates ingestion on `status`. Every page this read produced has
    text, so without the bound block the receipt would say `complete` for a document six pages of
    which were never opened -- a resource limit quietly becoming a data loss nobody looks for."""
    budget = ExtractionBudget(max_pages=4, time_budget_seconds=0)
    docs = list(SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget).lazy_load())

    receipt = _receipt(docs)
    assert receipt["status"] == "partial"
    assert receipt["extraction_bound"] == {
        "stopped_reason": "page_limit",
        "pages_read": 4,
        "pages_not_included": 6,
    }


def test_an_unbounded_document_has_no_bound_block(pdf_of_10_pages):
    receipt = _receipt(list(SafePyPDFLoader(pdf_of_10_pages).lazy_load()))
    assert receipt["status"] == "complete"
    assert "extraction_bound" not in receipt, (
        "absent means the whole document was read -- it must never mean 'we did not check'"
    )


def test_the_marker_is_found_wherever_it_sits(pdf_of_10_pages):
    """The receipt scans every document for the marker instead of trusting it to be on the last
    one. Which page carries it is a loader detail, and a receipt coupled to that detail would go
    quietly wrong the day it changed."""
    docs = [
        Document(page_content="page one", metadata={"page": 0}),
        Document(
            page_content="page two",
            metadata={
                "page": 1,
                STOPPED_KEY: "time_limit",
                ATTEMPTED_KEY: 2,
                NOT_INCLUDED_KEY: 97,
            },
        ),
        Document(page_content="page three", metadata={"page": 2}),
    ]
    receipt = _receipt(docs)
    assert receipt["status"] == "partial"
    assert receipt["extraction_bound"]["stopped_reason"] == "time_limit"
    assert receipt["extraction_bound"]["pages_not_included"] == 97


# =====================================================================================
# The four findings an independent review returned against the first version of this
# feature. Each test below fails against that version; three of them describe a way the
# bound could make a FALSE STATEMENT, which is the one thing this mechanism must not do.
# =====================================================================================


@pytest.fixture
def pdf_with_a_blank_cover(tmp_path):
    """Nine readable pages behind one page with no text layer -- a cover sheet, a title page,
    a blank leading scan. Utterly ordinary, and the case that breaks a naive page bound."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    for i in range(9):
        _text_page(writer, "Page %d marker" % (i + 2))
    path = os.path.join(str(tmp_path), "cover.pdf")
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


def test_a_bound_that_reads_only_blank_pages_blames_the_bound_not_the_file(
    pdf_with_a_blank_cover,
):
    """F1. The refusal is right; the REASON was a fabrication.

    With the bound set to one page and that page blank, nothing is extracted, so the
    empty-extraction guard refuses -- correctly, since there is nothing to store and a 200
    would be the fake success it exists to prevent. But it refused with "the file may be
    empty, image-only/scanned, corrupted, or password-protected": four accusations about a
    perfectly readable nine-page document, none of them true, and the real cause was our own
    setting. The control at the end of this test extracts those nine pages fine.
    """
    from fastapi import HTTPException

    from app.routes.document_routes import _assert_extractable_content

    docs = list(
        SafePyPDFLoader(
            pdf_with_a_blank_cover,
            extraction_budget=ExtractionBudget(max_pages=1, time_budget_seconds=0),
        ).lazy_load()
    )

    with pytest.raises(HTTPException) as raised:
        _assert_extractable_content(docs, "cover.pdf")

    detail = raised.value.detail
    message = detail["message"]
    assert raised.value.status_code == 422, "nothing was extracted, so nothing may be stored"
    assert detail["extraction"].get("extraction_bound") is not None, (
        "the cause must be machine-readable, not only in prose"
    )
    for accusation in ("corrupted", "password-protected", "image-only"):
        assert accusation not in message, (
            "the file is not what went wrong: %r" % message
        )
    assert "read bound" in message and "PDF_EXTRACT_MAX_PAGES" in message, (
        "the message must name the limit and the knob that moves it: %r" % message
    )

    # THE CONTROL. Without it this test would pass against a bound that broke the file.
    unbounded = list(SafePyPDFLoader(pdf_with_a_blank_cover).lazy_load())
    receipt = _assert_extractable_content(unbounded, "cover.pdf")
    assert receipt["units_extracted"] == 9, (
        "the same document extracts nine pages unbounded -- which is what makes the bounded "
        "refusal a statement about US, not about it"
    )


def test_the_bound_stops_the_producer_not_just_the_result(pdf_of_10_pages, monkeypatch):
    """F2. With image extraction on, the bound used to be applied to an already-materialised
    list: every page was parsed and allocated -- the exact exhaustion this exists to prevent --
    and the surplus was then thrown away. Measured: 10 pages parsed for a 3-page bound.

    Counted at the PRODUCER, because that is where the cost is. Asserting on the pages that come
    OUT cannot tell the two implementations apart -- both yield three.
    """
    import app.utils.document_loader as loader_module

    produced = []
    real_lazy_load = loader_module.PyPDFLoader.lazy_load

    def counting_lazy_load(self):
        for index, page in enumerate(real_lazy_load(self)):
            produced.append(index)
            yield page

    monkeypatch.setattr(loader_module.PyPDFLoader, "lazy_load", counting_lazy_load)

    loader = SafePyPDFLoader(
        pdf_of_10_pages,
        extract_images=True,
        extraction_budget=ExtractionBudget(max_pages=3, time_budget_seconds=0),
    )
    docs = list(loader.lazy_load())

    assert len(docs) == 3
    assert len(produced) <= 4, (
        "the producer parsed %d pages for a 3-page bound; the bound must wrap the generator, "
        "not filter its output" % len(produced)
    )


def test_a_spent_budget_fails_loudly_instead_of_reporting_an_empty_document(pdf_of_10_pages):
    """F7. A budget with nothing left yields no pages AND stamps nothing, so the receipt says
    `empty` with no `extraction_bound` at all -- the caller is told their readable file has no
    text, and nothing records that a limit did it. Unreachable on the live path, which is
    exactly why it has to be loud: the day it becomes reachable, a crash naming the cause is
    honest and a silent empty receipt is not.
    """
    budget = ExtractionBudget(max_pages=3, time_budget_seconds=0)
    loader = SafePyPDFLoader(pdf_of_10_pages, extraction_budget=budget)

    first = list(loader.lazy_load())
    assert len(first) == 3, "the first pass spends the budget"

    with pytest.raises(RuntimeError) as raised:
        list(loader.lazy_load())
    assert "already spent" in str(raised.value)
    assert "fresh ExtractionBudget" in str(raised.value), (
        "the error must say what to do, not only that something is wrong"
    )


def test_the_route_path_really_builds_a_bounded_loader(pdf_of_10_pages, monkeypatch):
    """F4. Every other test here constructs the budget by hand and drives the loader directly.
    Deleting the one line that wires the feature into `load_file_content` left the whole suite
    green -- 466 passed -- so nothing proved a real request ever got a bounded loader.

    This drives the function the embed routes actually call. It also exercises the only
    supported way to reconfigure at runtime (patching this module's own global), which is the
    narrower claim the budget's docstring now makes.
    """
    import asyncio

    import app.utils.extraction_budget as budget_module
    from app.routes.document_routes import load_file_content

    monkeypatch.setattr(budget_module, "PDF_EXTRACT_MAX_PAGES", 2)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        loaded = asyncio.run(
            load_file_content("ledger.pdf", "application/pdf", pdf_of_10_pages, executor)
        )
    docs = loaded[0]

    assert len(docs) == 2, (
        "the route path must build the loader WITH the configured budget; got %d pages"
        % len(docs)
    )
    marker = [d for d in docs if STOPPED_KEY in (d.metadata or {})]
    assert marker, "and the stop must be stamped, or the receipt cannot report it"
    assert marker[0].metadata[STOPPED_KEY] == "page_limit"

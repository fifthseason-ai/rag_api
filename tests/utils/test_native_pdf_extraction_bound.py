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
    NOT_ATTEMPTED_KEY,
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
    assert marker[NOT_ATTEMPTED_KEY] == 6, (
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
    """`pages_not_attempted` is None, never 0, when the file's own page count cannot be read.
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
    assert docs[-1].metadata[NOT_ATTEMPTED_KEY] is None, (
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
        "pages_not_attempted": 6,
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
                NOT_ATTEMPTED_KEY: 97,
            },
        ),
        Document(page_content="page three", metadata={"page": 2}),
    ]
    receipt = _receipt(docs)
    assert receipt["status"] == "partial"
    assert receipt["extraction_bound"]["stopped_reason"] == "time_limit"
    assert receipt["extraction_bound"]["pages_not_attempted"] == 97

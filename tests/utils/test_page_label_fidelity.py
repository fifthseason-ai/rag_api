"""`page_label` is the document's OWN page number, and a consumer now depends on it.

WHY THIS FILE EXISTS. Core renders `page_label` in preference to `page + 1`, because `page` is a
0-based physical index and the two disagree on any document with front matter. Until now nothing on
this side drove a real PDF with real page labels: both existing references to `page_label` in the
suite are hand-built `Document` objects with the value written in by hand. So a loader change that
dropped the key, or replaced it with `str(page + 1)`, would have passed every test here while
silently making every citation into a front-matter document point at the wrong page.

That is the failure this lane keeps meeting: a fixture asserting a value the pipeline was never
asked to produce.

MEASURED 2026-09-20 against a real service and a real pgvector, on a PDF with a `/PageLabels` tree
of lowercase roman front matter followed by arabic restarting at 1:

    physical page 0  ->  page_label "i"     `page + 1` would say 1
    physical page 1  ->  page_label "ii"    `page + 1` would say 2
    physical page 2  ->  page_label "1"     `page + 1` would say 3

The third row is the one that matters most and it is the least obvious. Front matter renders a
visibly wrong KIND of number and someone may notice. A body page the document prints as "1"
rendering as "p. 3" is entirely plausible, sends a reader to a page printed 3, and gives no sign
anything went wrong. Every body page of such a document is shifted by the front-matter count.
"""

import pytest

from app.utils.document_loader import get_loader

# Two roman front-matter pages, then the body restarting at arabic 1 -- the shape of essentially
# every report with a cover and a contents page.
PAGES = [
    ("FRONTMATTER-ALPHA cover sheet", 0, "i"),
    ("FRONTMATTER-BETA table of contents", 1, "ii"),
    ("BODY-GAMMA the first page of the actual report", 2, "1"),
    ("BODY-DELTA the second page of the actual report", 3, "2"),
]


@pytest.fixture
def labelled_pdf(tmp_path):
    """A PDF carrying a real `/PageLabels` tree, not a filename convention."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject

    from .test_scanned_pdf_ocr import _bytes, _text_page

    writer = pypdf.PdfWriter()
    for text, _idx, _label in PAGES:
        _text_page(writer, [text])

    roman = DictionaryObject()
    roman[NameObject("/S")] = NameObject("/r")
    arabic = DictionaryObject()
    arabic[NameObject("/S")] = NameObject("/D")
    arabic[NameObject("/St")] = NumberObject(1)
    labels = DictionaryObject()
    labels[NameObject("/Nums")] = ArrayObject(
        [NumberObject(0), roman, NumberObject(2), arabic]
    )
    writer._root_object[NameObject("/PageLabels")] = writer._add_object(labels)

    path = tmp_path / "annual report.pdf"
    with open(str(path), "wb") as fh:
        fh.write(_bytes(writer))

    loader, _known, _ext = get_loader("annual report.pdf", "application/pdf", str(path))
    docs = list(loader.load())
    by_marker = {}
    for doc in docs:
        marker = next((t.split()[0] for t, _i, _l in PAGES if t.split()[0] in doc.page_content),
                      None)
        if marker is not None:
            by_marker.setdefault(marker, doc)
    return docs, by_marker


def test_the_fixture_really_carries_labels_that_differ_from_the_index(labelled_pdf):
    """Precondition. If the labels happened to equal `page + 1`, every assertion below would be
    true of a file that cannot express the defect, and this file would be decorative."""
    _docs, by_marker = labelled_pdf
    diverging = [
        label for text, idx, label in PAGES
        if text.split()[0] in by_marker and label != str(idx + 1)
    ]
    assert len(diverging) >= 3, (
        "the fixture's labels do not diverge from page + 1 (%r), so it cannot express the defect "
        "this file exists for" % (diverging,)
    )


def test_every_page_carries_the_label_the_document_prints(labelled_pdf):
    _docs, by_marker = labelled_pdf
    for text, idx, label in PAGES:
        marker = text.split()[0]
        assert marker in by_marker, "page %d did not survive extraction" % idx
        meta = by_marker[marker].metadata
        assert meta.get("page") == idx, (
            "%s: expected physical index %d, got %r" % (marker, idx, meta.get("page")))
        assert meta.get("page_label") == label, (
            "%s: the document prints %r on this page; the loader reported %r. A citation built "
            "from this sends the reader to the wrong page." % (marker, label, meta.get("page_label")))


def test_the_label_is_not_arithmetic_on_the_index(labelled_pdf):
    """The property that makes the field load-bearing at all.

    If `page_label` were always `str(page + 1)` it would carry no information and a consumer
    could drop it safely. It is not, and dropping it is exactly the defect found in Core's route
    on 2026-09-20 -- so this asserts the divergence directly rather than leaving it implied by
    the values above.
    """
    _docs, by_marker = labelled_pdf
    wrong_if_computed = [
        (text.split()[0], label, str(idx + 1))
        for text, idx, label in PAGES
        if text.split()[0] in by_marker and label != str(idx + 1)
    ]
    assert wrong_if_computed, "no page diverges, so page_label carries nothing here"
    for marker, label, computed in wrong_if_computed:
        meta = by_marker[marker].metadata
        assert meta.get("page_label") != computed, (
            "%s: page_label is %r, which is page + 1. Either the loader stopped reading the "
            "document's own labels, or this fixture stopped carrying them." % (marker, computed))


def test_the_label_is_a_string_not_a_number(labelled_pdf):
    """Core's locator type declares `pageLabel` as a string, and "iii" is not a number.

    Cheap to assert and it pins the one field in the locator set that is deliberately NOT an int.
    A loader coercing it would break roman front matter specifically.
    """
    _docs, by_marker = labelled_pdf
    for text, _idx, _label in PAGES:
        marker = text.split()[0]
        if marker in by_marker:
            value = by_marker[marker].metadata.get("page_label")
            assert isinstance(value, str), (
                "%s: page_label is %r (%s); it must stay a string, because 'iii' is not a number"
                % (marker, value, type(value).__name__))

"""A unit's locator must survive chunk splitting (FILES-01 F03, A08).

When one source unit (a PDF page, a PPTX slide, an XLSX sheet element, a CSV row) carries
more text than CHUNK_SIZE, `_prepare_documents_sync` splits it into several stored chunks.
A08 requires a searched term to resolve to the exact unit it belongs to, so EVERY chunk of
that unit must keep the unit's locator — not just the first.

MEASURED 2026-09-21 (probe, image files01-ocr-test:wip): a 9600-char page (CHUNK_SIZE=1500)
became 7 chunks, all carrying page=7/page_label=vii. This pins that so a future change to
the split/stamp path cannot quietly keep the locator on only the first chunk (the mutation
below reddens exactly that regression). This coverage did not exist: prior tests proved
one-Document-per-unit (test_lazy_load CSV rows, test_parser_fitness XLSX elements) and
receipt<->chunk agreement (#58), but none split a single unit across chunks.
"""
from langchain_core.documents import Document

from app.config import CHUNK_SIZE
from app.routes.document_routes import _prepare_documents_sync

# Comfortably several chunks: >> CHUNK_SIZE, and long enough that overlap cannot collapse it.
_BIG = "Section text belonging to page seven. " * 400


def _prepare(meta):
    return _prepare_documents_sync(
        [Document(page_content=_BIG, metadata=dict(meta))],
        "fid", "userA", False, tenant_id="tenantA",
    )


def test_a_pdf_page_split_into_many_chunks_keeps_its_page_locator_on_every_chunk():
    docs = _prepare({"page": 7, "page_label": "vii", "source": "f.pdf"})
    assert len(docs) > 1, f"precondition: the unit must split (got {len(docs)} chunk)"
    assert all(d.metadata.get("page") == 7 for d in docs), [d.metadata.get("page") for d in docs]
    assert all(d.metadata.get("page_label") == "vii" for d in docs)
    # ids too: a chunk that lost its file_id/user_id would be unfilterable / unattributable.
    assert all(d.metadata.get("file_id") == "fid" for d in docs)
    assert all(d.metadata.get("user_id") == "userA" for d in docs)


def test_slide_and_sheet_and_row_locators_survive_split():
    for key, val in (("slide_number", 3), ("page_name", "Sales"), ("row", 42)):
        docs = _prepare({key: val})
        assert len(docs) > 1, (key, len(docs))
        assert all(d.metadata.get(key) == val for d in docs), (key, [d.metadata.get(key) for d in docs])


def test_two_adjacent_units_are_not_merged_into_one_locator():
    """Each input Document is split independently, so page 1's tail and page 2's head never
    share a chunk that then carries only one page number. (Whole-file loaders would already
    have lost the locator before this function; those are covered per-format elsewhere.)"""
    docs = _prepare_documents_sync(
        [Document(page_content=_BIG, metadata={"page": 1}),
         Document(page_content=_BIG, metadata={"page": 2})],
        "fid", "userA", False,
    )
    pages = {d.metadata.get("page") for d in docs}
    assert pages == {1, 2}
    # No chunk is missing a page (a merged/blank-locator chunk would show None).
    assert None not in pages

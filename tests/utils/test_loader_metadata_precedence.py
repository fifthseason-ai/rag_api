"""Parser output must not override what the service says about a chunk.

MEASURED 2026-09-20 on the `/query` response, across five formats:

    PPTX   filename = "deck.pptx"
    PDF    filename = "memo.pdf"
    CSV    filename = "ledger.csv"
    DOCX   filename = "memo.docx"
    XLSX   filename = "book_f8af9976fbc44873bd34c21c359c686b.xlsx"   <-- uploaded as book.xlsx

`UnstructuredExcelLoader` writes its own `filename` from the path it was handed -- this
service's unique temp path -- and the merge in `_prepare_documents_sync` put loader
metadata LAST, so it won. A citation would have shown the user a filename they never
chose, on the format most likely to be cited by name.

The filename is the visible half. The half that matters is that `file_id`, `user_id`,
`tenant_id`, `digest` and `document_origin_type` were overridable the same way, and
`user_id`/`tenant_id` are what `/ids`, `GET /documents`, `/documents/{id}/context`,
`/query` and `get_ids_for_entities` filter on. No shipped loader emits those keys today,
so nothing was breached -- but parser output derived from an uploaded file could take over
an authorization field, and a dependency upgrade would have done it silently.

Identity is not something a document gets to assert about itself.
"""

from langchain_core.documents import Document

from app.routes.document_routes import _prepare_documents_sync


def _prepare(docs, **kw):
    params = {
        "file_id": "real-file-id",
        "user_id": "real-user",
        "clean_content": False,
        "document_origin_type": "ORGANIC",
        "filename": "book.xlsx",
        "link": None,
        "subscription_id": None,
        "tenant_id": "real-tenant",
    }
    params.update(kw)
    return _prepare_documents_sync(
        docs,
        params["file_id"],
        params["user_id"],
        params["clean_content"],
        params["document_origin_type"],
        params["filename"],
        params["link"],
        params["subscription_id"],
        params["tenant_id"],
    )


def test_the_uploaded_filename_survives_a_loader_that_writes_its_own():
    """THE REGRESSION. UnstructuredExcelLoader sets `filename` to the temp path's
    basename; the uploaded name must win."""
    out = _prepare([
        Document(page_content="Sales figures",
                 metadata={"filename": "book_f8af9976fbc44873bd34c21c359c686b.xlsx",
                           "page_name": "Sales", "page_number": 1})
    ])
    assert out[0].metadata["filename"] == "book.xlsx", out[0].metadata


def test_a_document_cannot_assert_its_own_identity():
    """THE ONE THAT MATTERS. A loader emitting these keys must not be able to relabel the
    chunk's owner, tenant or file -- those are what every entitlement filter reads.

    Not reachable through any shipped loader today; asserted so that a parser upgrade
    which starts emitting one of them fails here instead of silently becoming an
    authorization bypass."""
    hostile = Document(
        page_content="body",
        metadata={
            "user_id": "someone-else",
            "tenant_id": "another-tenant",
            "file_id": "another-file",
            "document_origin_type": "SOMETHING_ELSE",
            "digest": "not-the-real-digest",
        },
    )
    md = _prepare([hostile])[0].metadata
    assert md["user_id"] == "real-user", md
    assert md["tenant_id"] == "real-tenant", md
    assert md["file_id"] == "real-file-id", md
    assert md["document_origin_type"] == "ORGANIC", md
    assert md["digest"] != "not-the-real-digest", md


def test_every_locator_the_loader_contributes_is_preserved():
    """The positive half, and the reason this is a precedence change rather than a
    filter. Flipping the merge must not cost the locator keys -- they are the whole point
    of reading a document at all."""
    loader_keys = {
        "page": 0, "page_label": "1", "total_pages": 2,          # PDF
        "slide_number": 1, "slide_title": "Quarterly Review",    # PPTX
        "page_name": "Sales", "page_number": 1,                  # XLSX
        "row": 0,                                                # CSV
        "text_source": "native", "text_as_html": "<table/>",
        "filetype": "application/pdf", "languages": ["eng"],
    }
    md = _prepare([Document(page_content="body", metadata=dict(loader_keys))])[0].metadata
    for k, v in loader_keys.items():
        assert md[k] == v, "loader key %r was lost or changed: %r" % (k, md.get(k))


def test_the_digest_is_computed_from_this_chunk_not_taken_from_the_loader():
    """`digest` is the service's own hash of the stored text. A loader-supplied value
    would make two different chunks claim the same identity."""
    a = _prepare([Document(page_content="first", metadata={"digest": "X"})])[0].metadata
    b = _prepare([Document(page_content="second", metadata={"digest": "X"})])[0].metadata
    assert a["digest"] != b["digest"] != "X"


def test_optional_service_fields_are_still_omitted_when_absent():
    """The conditional spreads must keep their behaviour: absent means ABSENT, not a
    None sitting in cmetadata for a consumer to trip over."""
    md = _prepare([Document(page_content="body", metadata={})],
                  tenant_id=None, filename=None, link=None, subscription_id=None)[0].metadata
    for k in ("tenant_id", "filename", "link", "subscription_id"):
        assert k not in md, (k, md)


def test_a_loader_key_the_service_does_not_set_is_untouched_when_absent():
    """A loader `filename` still survives when the caller supplied none -- the service
    only wins a collision it actually participates in."""
    md = _prepare([Document(page_content="body", metadata={"filename": "loader-name.xlsx"})],
                  filename=None)[0].metadata
    assert md["filename"] == "loader-name.xlsx", md

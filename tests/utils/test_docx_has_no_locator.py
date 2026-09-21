"""DOCX must never carry a per-unit locator key (FILES-DEV F-DOCX1).

`app.routes.document_routes._UNIT_LOCATOR_KEYS` is the single place that decides
which metadata key means "this chunk can be cited at a page / slide / sheet":

    (("page", "page"), ("slide", "slide_number"), ("sheet", "page_name"))

DOCX has no such unit. `Docx2txtLoader` flattens the whole document to ONE
Document whose metadata is exactly `{'source': ...}` (measured, both routings),
so every DOCX folds into a single `none` unit with nothing finer to cite. If a
locator key ever appeared on a DOCX Document, two things would follow, and the
existing coverage catches only the first:

  1. `_extraction_receipt` would report a `locator_kind` and list
     `empty_locators` for a document that has no pages at all -- already
     asserted from the 200 body by
     `tests/utils/test_extraction_status.py::test_embed_docx_reports_complete_locator_none_and_writes`.

  2. `_prepare_documents_sync` splices the loader's metadata into EVERY stored
     chunk (`**(doc.metadata or {})`), so the fabricated locator would be
     PERSISTED and handed to whatever renders a citation. Nothing asserted that
     surface, and the receipt cannot speak for it: the receipt is recomputed per
     request, the chunk metadata is what outlives it.

The receipt is also not a sufficient proxy for the rule, which is why these
tests assert KEY ABSENCE rather than reading `locator_kind`. The detection loop
tests `.get(key) is not None`, so a locator key present with value `None` leaves
`locator_kind == "none"` while the key still propagates into stored chunks --
a receipt that says "no locator" over rows that carry one. That measured gap is
pinned below in `test_receipt_alone_cannot_prove_the_rule`.

Assertions are bound to `_UNIT_LOCATOR_KEYS` itself, not to a copied list of key
names, so a locator family added later is automatically checked against DOCX.

All fixtures are SYNTHETIC and generated at test time (SYN-KNOWLEDGE-01 label);
no client content.
"""

import zipfile

import pytest
from langchain_core.documents import Document

from app.routes.document_routes import (
    _UNIT_LOCATOR_KEYS,
    _extraction_receipt,
    _prepare_documents_sync,
)
from app.utils.document_loader import get_loader

# The synthetic DOCX already proven to extract heading/body/table/header/footer.
from tests.utils.test_parser_fitness import W_NS, make_docx

_DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

#: Both routings in `get_loader`'s doc/docx branch reach `Docx2txtLoader`: the
#: OOXML content type and the legacy `application/msword` one. A rule proven on
#: one branch says nothing about the other.
_DOCX_ROUTINGS = [
    ("report.docx", _DOCX_MIME),
    ("report.doc", "application/msword"),
]


def make_long_docx(path, paragraphs=40):
    """A DOCX whose body is long enough to split into MORE THAN ONE chunk.

    Needed because `_prepare_documents_sync` copies the loader metadata onto
    every chunk: a one-chunk fixture cannot express "the locator leaked into
    chunk 7", so the absence assertion would be weaker than it looks.
    """
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="'
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    body = "".join(
        "<w:p><w:r><w:t>"
        f"SYN-KNOWLEDGE-01 paragraph {i:03d}. "
        "This sentence exists only to push the extracted text past the splitter "
        "boundary so that more than one stored chunk is produced."
        "</w:t></w:r></w:p>"
        for i in range(paragraphs)
    )
    document = f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def _locator_keys_present(metadata):
    """Locator keys PRESENT in `metadata`, regardless of their value.

    Presence, not truthiness: a key stamped with `None` is still a key that
    reaches a citation consumer, and it is the exact case `locator_kind` cannot
    see.
    """
    meta = metadata or {}
    return sorted(key for _kind, key in _UNIT_LOCATOR_KEYS if key in meta)


def _assert_no_locator_keys(metadatas, where):
    offenders = [
        (i, _locator_keys_present(m))
        for i, m in enumerate(metadatas)
        if _locator_keys_present(m)
    ]
    assert not offenders, (
        f"DOCX must carry no per-unit locator key, but {where} stamped "
        f"{offenders} (locator keys: {[k for _k, k in _UNIT_LOCATOR_KEYS]})"
    )


# ===========================================================================
# The rule, at the real loader
# ===========================================================================


@pytest.mark.parametrize("filename,content_type", _DOCX_ROUTINGS)
def test_docx_loader_emits_no_locator_key(tmp_path, filename, content_type):
    """The REAL loader `get_loader` selects for DOCX emits no locator key.

    Driven through `get_loader` rather than a hand-built Document list, so a
    change of DOCX loader (or of the routing) is what this test reads.
    """
    path = tmp_path / "report.docx"
    make_docx(str(path))

    loader, known_type, _ext = get_loader(filename, content_type, str(path))
    docs = loader.load()

    assert known_type is True
    assert docs, "fixture produced no Documents -- the case cannot be expressed"
    # Guard the premise: text really was extracted, so this is a loaded DOCX and
    # not an empty read that trivially carries no metadata at all.
    assert any(d.page_content.strip() for d in docs)

    _assert_no_locator_keys([d.metadata for d in docs], f"{filename} loader output")


@pytest.mark.parametrize("filename,content_type", _DOCX_ROUTINGS)
def test_docx_receipt_reports_none_locator_and_no_locators(
    tmp_path, filename, content_type
):
    """Acceptance, at the receipt built from REAL loader output: `locator_kind`
    stays 'none', the whole document folds into one unit, and nothing is listed
    as a locator."""
    path = tmp_path / "report.docx"
    make_docx(str(path))

    loader, _known, _ext = get_loader(filename, content_type, str(path))
    receipt = _extraction_receipt(loader.load())

    assert receipt["locator_kind"] == "none"
    assert receipt["units_total"] == 1
    assert receipt["units_extracted"] == 1
    assert receipt["empty_locators"] == []
    assert receipt["reasons"] == []


# ===========================================================================
# The rule, at the surface that OUTLIVES the request: stored chunk metadata
# ===========================================================================


def test_docx_stored_chunk_metadata_has_no_locator_key(tmp_path):
    """`_prepare_documents_sync` splices loader metadata into every chunk it
    persists. A locator stamped upstream would be stored on all of them and read
    back as a citation, long after the receipt that would have reported it is
    gone."""
    path = tmp_path / "long.docx"
    make_long_docx(str(path))

    loader, _known, _ext = get_loader("long.docx", _DOCX_MIME, str(path))
    docs = loader.load()

    prepared = _prepare_documents_sync(
        docs,
        file_id="syn-file-id",
        user_id="syn-user",
        clean_content=True,
        filename="long.docx",
    )

    # The fixture must actually reach the multi-chunk case it claims to cover.
    assert len(prepared) > 1, (
        f"fixture produced {len(prepared)} chunk(s); the per-chunk propagation "
        "case is not exercised"
    )
    _assert_no_locator_keys(
        [d.metadata for d in prepared], "stored DOCX chunk metadata"
    )
    # Proves the metadata pathway under test is live: the loader's own key DID
    # propagate to every chunk, so an added locator key would have too.
    assert all("source" in d.metadata for d in prepared)


# ===========================================================================
# Controls -- these prove the checks above can fail
# ===========================================================================


def test_locator_absence_check_rejects_a_stamped_locator():
    """Positive control for the mechanism, not the term: `_assert_no_locator_keys`
    must FAIL on each locator family, and must fail on a `None`-valued key too."""
    for _kind, key in _UNIT_LOCATOR_KEYS:
        for value in (0, 1, "Sheet1", None):
            with pytest.raises(AssertionError):
                _assert_no_locator_keys([{"source": "x", key: value}], "control")

    # ...and passes on metadata that carries no locator family at all.
    _assert_no_locator_keys([{"source": "x", "filename": "a.docx"}], "control")


def test_receipt_alone_cannot_prove_the_rule():
    """Why these tests assert key ABSENCE instead of reading `locator_kind`.

    The detection loop asks `.get(key) is not None`, so a locator key stamped
    with `None` leaves the receipt saying 'none' while the key still rides into
    every stored chunk. Pinned as measured behaviour: if this ever changes, the
    reasoning in this module's docstring needs rereading, not silent drift."""
    doc = Document(page_content="whole document body", metadata={"source": "x", "page": None})

    assert _extraction_receipt([doc])["locator_kind"] == "none"  # receipt sees nothing
    with pytest.raises(AssertionError):  # the key is nonetheless there
        _assert_no_locator_keys([doc.metadata], "control")

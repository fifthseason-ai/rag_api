"""DOCX carries a per-unit locator family from E1 onward (was: none).

RED-FIRST GATE FOR E1. Until E1, DOCX had no per-unit locator: `Docx2txtLoader`
flattened the whole document to ONE Document whose metadata was exactly
`{'source': ...}`, so every DOCX folded into a single `none` unit (F-DOCX1). This
file was the gate that pinned that absence. E1 adds source-location fidelity: a
block-indexed `SafeDocxLoader` emits one Document per authored unit (heading /
paragraph / table cell / header / footer) carrying the DOCX per-unit locator, and
registers ONE additive family in `_UNIT_LOCATOR_KEYS`.

So the three DOCX-absence assertions this file used to make are FLIPPED here from
"asserts none" to "asserts the new family", each replaced by an assertion that is
equally specific about the family (per the FILES lead's E1 agreement,
2026-09-23). The function names are kept verbatim so nothing is deleted and the
gate stays traceable; their bodies now assert PRESENCE. The three flipped:

  1. `test_docx_loader_emits_no_locator_key` -- the loader now stamps the family
     on every emitted unit.
  2. `test_docx_receipt_reports_none_locator_and_no_locators` -- the receipt now
     reports the DOCX family (not `none`); it is the only place `locator_kind`
     for DOCX is asserted, so its replacement asserts the new family just as
     specifically, and it is NOT deleted.
  3. `test_docx_stored_chunk_metadata_has_no_locator_key` -- the family reaches
     EVERY STORED chunk through `_prepare_documents_sync`, not just the loader
     Document. This is the "a registered tuple entry that nothing stamps changes
     nothing" proof (FILES lead caution): the stamp is proven on the persisted
     surface that outlives the request, in-process (no pgvector).

VALUE/KEY ARE PENDING. The locator_kind vocabulary value and the cmetadata key
are the FILES lead's decision and are NOT hardcoded here: every assertion reads
`SafeDocxLoader._DOCX_LOCATOR_KIND` / `._DOCX_LOCATOR_KEY`, so the ruling is a
one-token change in the loader. No literal of the value appears in this file.

The CONTROLS are unchanged and still green: `_assert_no_locator_keys` still fires
on any registered key (now including the DOCX family), and
`test_receipt_alone_cannot_prove_the_rule` still shows a `None`-valued key is
invisible to `locator_kind` yet rides into chunks -- which is why the presence
assertions read the KEY, not only `locator_kind`.

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
from app.utils.document_loader import SafeDocxLoader, get_loader

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


#: The DOCX per-unit locator family, read from the loader so the PENDING value/key
#: ruling is a one-token change and no literal of the value appears here.
_DOCX_KEY = SafeDocxLoader._DOCX_LOCATOR_KEY
_DOCX_KIND = SafeDocxLoader._DOCX_LOCATOR_KIND


def _assert_docx_family_on_every_unit(metadatas, where):
    """Every emitted/stored unit must carry the DOCX locator key with an int
    index, and the family must be registered in `_UNIT_LOCATOR_KEYS`."""
    assert any(k == _DOCX_KEY for _kind, k in _UNIT_LOCATOR_KEYS), (
        f"the DOCX family key {_DOCX_KEY!r} is not registered in _UNIT_LOCATOR_KEYS"
    )
    missing = [
        (i, dict(m or {}))
        for i, m in enumerate(metadatas)
        if not isinstance((m or {}).get(_DOCX_KEY), int)
    ]
    assert not missing, (
        f"{where}: every DOCX unit must carry an int {_DOCX_KEY!r}, but these did "
        f"not: {missing}"
    )


@pytest.mark.parametrize("filename,content_type", _DOCX_ROUTINGS)
def test_docx_loader_emits_no_locator_key(tmp_path, filename, content_type):
    """FLIPPED for E1 (name kept): the REAL loader `get_loader` selects for DOCX now
    stamps the per-unit locator family on EVERY emitted unit, block-indexed 0-based
    and contiguous in reading order.

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
    # The fixture has several blocks, so this is not a one-unit doc masquerading
    # as block-indexed.
    assert len(docs) > 1, f"{filename}: expected several block units, got {len(docs)}"

    _assert_docx_family_on_every_unit(
        [d.metadata for d in docs], f"{filename} loader output"
    )
    # Indices are 0-based and contiguous in emission (reading) order.
    assert [d.metadata[_DOCX_KEY] for d in docs] == list(range(len(docs)))
    # Provenance stays the uploaded file, never a working copy.
    assert all(d.metadata.get("source") == str(path) for d in docs)


@pytest.mark.parametrize("filename,content_type", _DOCX_ROUTINGS)
def test_docx_receipt_reports_none_locator_and_no_locators(
    tmp_path, filename, content_type
):
    """FLIPPED for E1 (name kept): the receipt built from REAL loader output now
    reports the DOCX family (not 'none'), one unit per emitted block, and every
    block-bearing unit has extractable text so nothing is listed empty. This is the
    only place `locator_kind` for DOCX is asserted, so the replacement pins the new
    family just as specifically as the old assertion pinned 'none'."""
    path = tmp_path / "report.docx"
    make_docx(str(path))

    loader, _known, _ext = get_loader(filename, content_type, str(path))
    docs = loader.load()
    receipt = _extraction_receipt(docs)

    assert receipt["locator_kind"] == _DOCX_KIND
    assert receipt["units_total"] == len(docs) > 1
    assert receipt["units_extracted"] == receipt["units_total"]
    assert receipt["empty_locators"] == []
    assert receipt["reasons"] == []


# ===========================================================================
# The rule, at the surface that OUTLIVES the request: stored chunk metadata
# ===========================================================================


def test_docx_stored_chunk_metadata_has_no_locator_key(tmp_path):
    """FLIPPED for E1 (name kept): the family reaches the STORED chunk, not just the
    loader Document. `_prepare_documents_sync` splices loader metadata into every
    chunk it persists (`**(doc.metadata or {})`), so the block index rides into
    every stored chunk and is read back as a citation after the per-request receipt
    is gone. This is the "a registered tuple entry that nothing stamps changes
    nothing" proof, on the persisted surface, in-process (no pgvector)."""
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

    # The fixture must actually reach the multi-chunk case it claims to cover: 40
    # paragraphs become 40 block-indexed units and many stored chunks, so the
    # per-chunk propagation of the family is exercised, not assumed.
    assert len(prepared) > 1, (
        f"fixture produced {len(prepared)} chunk(s); the per-chunk propagation "
        "case is not exercised"
    )
    _assert_docx_family_on_every_unit(
        [d.metadata for d in prepared], "stored DOCX chunk metadata"
    )
    # The service fields still win and provenance survives, alongside the family.
    assert all(d.metadata.get("source") == str(path) for d in prepared)
    assert all(d.metadata.get("file_id") == "syn-file-id" for d in prepared)


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

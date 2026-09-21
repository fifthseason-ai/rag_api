"""A legacy `.doc` gets a verdict instead of a zip error (FILES-DEV F-LEGACY3).

`.doc` had never been through this service. `get_loader` routes `doc` and `docx` down ONE branch to
`Docx2txtLoader`, and `docx2txt` reads an OOXML package -- a ZIP. A Word 97 `.doc` is an OLE2
compound file, so that half of the branch could only ever fail. Measured on 36d4fb6 with the genuine
fixture below, before the fix:

    get_loader -> Docx2txtLoader  known_type=True  ext=doc
    loader.load() -> zipfile.BadZipFile: File is not a zip file
    POST /embed  -> 400  "'legacy.doc' could not be read. The cause is not established -- it may be
                          the file or this service."   [attribution=undetermined]  rows=0

Nothing was stored, so this is NOT the garbage-extraction defect SP-01.6a closed. It is the other
half of that charter: we can tell exactly what this file is, and we said we could not tell.

It is also the gap PR #42 left open in writing. #42 turns a missing-LibreOffice failure into an
actionable 400, but it fires on `OSError("soffice command was not found")`, and a `.doc` never
reaches soffice -- the Word branch claims it first and dies in `zipfile`. #42's own comment records
that `.doc` "was never tested, so it is not claimed either". This suite tests it.

THE FIXTURE IS GENUINE, and that is the whole reason this item existed. It is a real Word 97 binary
produced by LibreOffice Writer 25.2.3.2 --

    soffice --headless --convert-to doc:"MS Word 97" --outdir /out src.txt

-- captured byte-for-byte and stored zlib+base64 so it can be rebuilt at test time, because this
tree keeps no binary fixtures and the test image has no LibreOffice. The provenance test below
re-proves it from the bytes on every run: a synthetic stand-in, or a corrupted literal, fails there
rather than silently weakening every assertion under it.

A previous pass concluded a genuine `.doc` could not be obtained at all -- LibreOffice refusing
`.txt`, `.rtf` and a hand-built `.docx` with "source file could not be loaded". The cause was not
the fixtures: `libreoffice-writer` was not installed in the conversion image (calc, draw and impress
were, which is why the `.ppt` and `.xlsx` controls passed and looked like proof the binary was fine).
"""

import base64
import datetime
import io
import os
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import (
    OLE2_MAGIC,
    UnsupportedDocumentError,
    get_loader,
)

_SECRET = "testsecret"

#: A genuine Word 97 document, zlib+base64. See the module docstring for its provenance.
_LEGACY_DOC_B64 = (
    "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"
    "wuzRUUT/z4HMNs7hrXz//Rv4a/ZX"
)


def write_legacy_doc(path) -> bytes:
    """Rebuild the genuine `.doc` at test time and return its bytes."""
    data = zlib.decompress(base64.b64decode("".join(_LEGACY_DOC_B64)))
    with open(str(path), "wb") as f:
        f.write(data)
    return data


def make_docx(path) -> None:
    """A minimal OOXML package -- the MODERN format, which must keep working untouched."""
    with zipfile.ZipFile(str(path), "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/'
            'content-types"><Default Extension="xml" ContentType="application/xml"/><Override '
            'PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocu'
            'ment.wordprocessingml.document.main+xml"/></Types>',
        )
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordproc'
            'essingml/2006/main"><w:body><w:p><w:r><w:t>Revenue grew twelve percent across EMEA.'
            "</w:t></w:r></w:p></w:body></w:document>",
        )


# ===========================================================================
# The fixture itself must be the real thing
# ===========================================================================


def test_the_fixture_is_a_genuine_word_97_binary(tmp_path):
    """Guards every other test in this file. A `.doc` cannot be synthesised the way this suite
    builds its PNGs and ZIPs, so the bytes are carried rather than constructed -- which means their
    provenance has to be re-proved from the bytes, not asserted in a comment.

    `olefile` ships in the test image (msoffcrypto depends on it), so the container can be opened
    and its streams named. A Word 97 document is an OLE2 compound file carrying a `WordDocument`
    stream; an `.xls` carries `Workbook` and a `.ppt` carries `PowerPoint Document`, so this
    distinguishes Word from its siblings rather than merely proving "some Office binary".
    """
    olefile = pytest.importorskip("olefile")

    path = tmp_path / "legacy.doc"
    data = write_legacy_doc(path)

    assert data[:8] == OLE2_MAGIC, "not an OLE2 compound file"
    assert data[:2] != b"PK", "a ZIP would mean this is secretly a .docx"

    ole = olefile.OleFileIO(str(path))
    streams = {"/".join(s) for s in ole.listdir()}
    assert "WordDocument" in streams, streams
    assert "Workbook" not in streams, "that would make this an .xls, not a .doc"
    assert len(data) > 4096, "a real Word 97 document is not a few hundred bytes"


# ===========================================================================
# The verdict
# ===========================================================================


def test_a_legacy_doc_is_refused_with_an_unsupported_verdict(tmp_path):
    """RED before this increment: `get_loader` returned `Docx2txtLoader` and `load()` raised
    `zipfile.BadZipFile`."""
    path = tmp_path / "legacy.doc"
    write_legacy_doc(path)

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader("legacy.doc", "application/msword", str(path))

    assert exc.value.verdict == "unsupported"


def test_the_refusal_names_the_format_and_the_action(tmp_path):
    path = tmp_path / "legacy.doc"
    write_legacy_doc(path)

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader("legacy.doc", "application/msword", str(path))

    message = str(exc.value).lower()
    assert "legacy word" in message
    assert ".doc" in message
    assert ".docx" in message        # what WOULD work
    assert "upload" in message       # ...and that they should send it
    assert "legacy.doc" in str(exc.value)  # names THEIR file


def test_the_refusal_does_not_claim_the_libreoffice_operator_fix(tmp_path):
    """The deliberate NON-claim, and the reason this is a separate test rather than a comment.

    #42's message offers "installing LibreOffice is the operator fix", which is true for the `.ppt`
    path that actually shells out to soffice. It is NOT true here: `Docx2txtLoader` would still be
    handed the same OLE2 bytes with LibreOffice installed, so the file would still fail. #42 already
    removed one false family claim after measurement contradicted it; this asserts we do not re-add
    it by copying a neighbouring format's wording.
    """
    path = tmp_path / "legacy.doc"
    write_legacy_doc(path)

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader("legacy.doc", "application/msword", str(path))

    message = str(exc.value).lower()
    assert "libreoffice" not in message
    assert "operator" not in message
    assert "retry" not in message and "retrying" not in message


def test_a_legacy_doc_no_longer_raises_badzipfile(tmp_path):
    """The exact original failure, pinned. `BadZipFile` is a `ValueError`, not a verdict, and it
    named our zip machinery to someone holding a Word file."""
    path = tmp_path / "legacy.doc"
    write_legacy_doc(path)

    with pytest.raises(UnsupportedDocumentError):
        loader, _, _ = get_loader("legacy.doc", "application/msword", str(path))
        loader.load()


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("legacy.doc", "application/msword"),
        ("legacy.doc", "application/octet-stream"),     # wrong type, right extension
        ("legacy.DOC", "application/msword"),           # extension case is lowered
        ("no_extension_at_all", "application/msword"),  # claimed by content type alone
    ],
)
def test_every_way_a_legacy_doc_reaches_the_word_branch_is_refused(
    tmp_path, filename, content_type
):
    """The branch is entered by extension OR content type, so both arms need the guard -- otherwise
    the refusal holds for the tidy case and the messy one still reaches `docx2txt`."""
    path = tmp_path / "f.bin"
    write_legacy_doc(path)

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader(filename, content_type, str(path))

    assert exc.value.verdict == "unsupported"


# ===========================================================================
# The negative space: what must NOT change
# ===========================================================================


def test_a_modern_docx_still_loads(tmp_path):
    """THE control. `.doc` and `.docx` share the branch this guard sits in, so the cost of refusing
    one is breaking the other -- and `.docx` is the format the refusal tells people to send."""
    path = tmp_path / "modern.docx"
    make_docx(path)

    loader, known_type, ext = get_loader(
        "modern.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        str(path),
    )

    assert known_type is True
    assert ext == "docx"
    assert "Revenue grew twelve percent" in " ".join(d.page_content for d in loader.load())


def test_a_docx_misnamed_dot_doc_still_loads(tmp_path):
    """The guard reads the CONTAINER, not the extension. Someone who renamed a modern document to
    `.doc` has a file we can read perfectly, and refusing it would be a new false negative of
    exactly the kind this module's review history is full of."""
    path = tmp_path / "actually_modern.doc"
    make_docx(path)

    loader, _, _ = get_loader("actually_modern.doc", "application/msword", str(path))

    assert "Revenue grew twelve percent" in " ".join(d.page_content for d in loader.load())


def test_a_legacy_xls_is_untouched_by_the_word_guard(tmp_path):
    """`.xls` is OLE2 too, and it reaches a DIFFERENT branch with its own encrypted-vs-corrupt
    verdict. A guard that fired on "OLE2" alone rather than "OLE2 on the Word branch" would steal
    that case and relabel a workbook as a Word document."""
    path = tmp_path / "legacy.xls"
    write_legacy_doc(path)  # OLE2 bytes are all the Word guard would look at

    try:
        get_loader("legacy.xls", "application/vnd.ms-excel", str(path))
    except UnsupportedDocumentError as e:  # pragma: no cover - the failure this test exists for
        pytest.fail("the Word guard reached the .xls branch: %s" % e)
    except Exception:
        pass  # the .xls branch's own verdict, whatever it is, is not this test's business


# ===========================================================================
# Route level: the verdict reaches the caller and nothing is stored
# ===========================================================================


def _hdr(ent, act, tid="tenantA", uid="testuser"):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ent,
        "act": act,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture()
def client(monkeypatch):
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    added = []

    async def recording_aadd(self, docs, ids=None, executor=None):
        added.append(list(docs))
        return ids

    async def dummy_delete(self, ids=None, collection_only=False, user_id=None,
                           document_origin_type=None, subscription_id=None, executor=None, **_):
        return None

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", recording_aadd)
    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)

    test_client = TestClient(app)
    test_client.inserted_batches = added  # type: ignore[attr-defined]
    return test_client


def test_embed_answers_422_unsupported_for_a_legacy_doc(client, tmp_path):
    """Measured before: 400 with "the cause is not established". The status change is the point --
    422 with a verdict is a decision about the FILE, 400 with an undetermined attribution is the
    service declining to say."""
    path = tmp_path / "legacy.doc"
    data = write_legacy_doc(path)

    r = client.post(
        "/embed",
        data={"file_id": "f-legacy-doc", "entity_id": "userA"},
        files={"file": ("legacy.doc", io.BytesIO(data), "application/msword")},
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["extraction"]["verdict"] == "unsupported"
    assert detail["extraction"]["status"] == "unsupported"
    assert "legacy.doc" in detail["message"]
    assert client.inserted_batches == []


def test_the_embed_response_no_longer_says_the_cause_is_not_established(client, tmp_path):
    """The original user-visible symptom, pinned as its own test so a regression reads as the
    sentence the uploader actually saw rather than as a status-code mismatch."""
    path = tmp_path / "legacy.doc"
    data = write_legacy_doc(path)

    r = client.post(
        "/embed",
        data={"file_id": "f-legacy-doc-2", "entity_id": "userA"},
        files={"file": ("legacy.doc", io.BytesIO(data), "application/msword")},
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert "cause is not established" not in r.text
    assert "zip" not in r.text.lower()


def test_text_route_also_refuses_a_legacy_doc(client, tmp_path):
    """`/text` loads through the same `load_file_content` seam; proving one route proves the seam,
    not the other routes."""
    path = tmp_path / "legacy.doc"
    data = write_legacy_doc(path)

    r = client.post(
        "/text",
        data={"file_id": "f-legacy-doc-text", "entity_id": "userA"},
        files={"file": ("legacy.doc", io.BytesIO(data), "application/msword")},
        headers=_hdr(ent=["userA"], act=["read"]),
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["extraction"]["verdict"] == "unsupported"
    assert client.inserted_batches == []


def test_embed_still_accepts_a_modern_docx(client, tmp_path):
    """Route-level control: the refusal did not cost us the format we tell people to send."""
    path = tmp_path / "modern.docx"
    make_docx(path)

    r = client.post(
        "/embed",
        data={"file_id": "f-modern-docx", "entity_id": "userA"},
        files={
            "file": (
                "modern.docx",
                io.BytesIO(path.read_bytes()),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert r.status_code == 200, r.text
    assert sum(len(b) for b in client.inserted_batches) >= 1

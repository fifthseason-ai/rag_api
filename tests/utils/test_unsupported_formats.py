"""Truthful verdicts for formats this service cannot read (KI-02 SP-01.6a).

`get_loader`'s final branch handed ANY unrecognised file to `TextLoader` with `known_type=False`, and
nothing downstream refuses on that flag — it is only reported in the response. So the bytes of a file we
have no extractor for were read as prose. Reproduced before the fix:

    ZIP   loader=TextLoader  known_type=False  docs=1  receipt=complete  survives_guard=True
          sample='PK\\x03\\x04\\x14·Q/]*M¾¿Ü\\x05Ü\\x05\\tinner.txthello from inside the archive hel'
    PNG   RAISED RuntimeError -> Could not detect encoding for /tmp/.../photo.png

The ZIP is the serious one: container framing plus fragments of its members, stored with an extraction
receipt reading **complete**. That is a garbage extraction counted as a success — the sibling of the
empty-extraction defect WP-C closed — and the uploader was told nothing. The PNG failed differently but
no better, naming our internals rather than their problem.

Both now get an `unsupported` verdict that names the format and says what to upload instead, surfaced as
a 422 by the same `load_file_content` seam every embed/text route loads through (SP-01.5).

The other half of this suite matters just as much: proving the refusal did NOT become over-eager. A
genuine text file must still parse, whatever its extension — including non-Latin text, where every byte
of the UTF-8 encoding is >= 0x80 and a printable-byte heuristic would have refused it outright.

Fixtures are SYNTHETIC and built at test time. The vector store is SIMULATED (`aadd_documents` recorded)
so "no rows written" is asserted against real insert attempts.
"""

import datetime
import io
import os
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import (
    EncryptedDocumentError,
    SheetExcelLoader,
    UnsupportedDocumentError,
    _OLE2_MAGIC,
    describe_unsupported_binary,
    get_loader,
    looks_like_binary,
)

_SECRET = "testsecret"

#: A GENUINE legacy Word-97/Excel-97 workbook (an OLE2 compound file), 5632 bytes, produced by
#: LibreOffice and audited to carry no author/host/personal metadata — only Calc boilerplate and a
#: synthetic content marker. It is the one fixture in this suite that is COPIED rather than built at
#: test time: openpyxl (and every library in the test image) writes the .xlsx ZIP format, not the
#: OLE2 .xls binary, so a genuinely-PARSING legacy .xls cannot be generated here. It exists so the
#: control below proves a real .xls keeps PARSING under this change, not merely that it routes.
_LEGACY_XLS = os.path.join(os.path.dirname(__file__), "fixtures", "legacy.xls")


# ===========================================================================
# Synthetic binaries
# ===========================================================================


def make_png(path):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x7f" * 8 for _ in range(8))
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


def make_zip(path):
    """A ZIP whose member is plain text — the case that used to report `complete`."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("inner.txt", "hello from inside the archive " * 50)


def make_mp3(path):
    with open(path, "wb") as f:
        f.write(b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(256)) * 20)


def make_exe(path):
    with open(path, "wb") as f:
        f.write(b"MZ\x90\x00" + bytes(range(256)) * 30)


def make_wav(path):
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 2048) + b"WAVEfmt " + bytes(2048))


BINARY_CASES = [
    ("photo.png", "image/png", make_png, "PNG image"),
    ("archive.zip", "application/zip", make_zip, "ZIP archive"),
    ("clip.mp3", "audio/mpeg", make_mp3, "MP3 audio"),
    ("app.exe", "application/octet-stream", make_exe, "Windows executable"),
    ("sound.wav", "audio/wav", make_wav, "WAV audio"),
]


# ===========================================================================
# The refusal
# ===========================================================================


@pytest.mark.parametrize("filename,mime,build,described", BINARY_CASES)
def test_unsupported_binary_is_refused_and_named(tmp_path, filename, mime, build, described):
    path = tmp_path / filename
    build(str(path))

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader(filename, mime, str(path))

    assert exc.value.verdict == "unsupported"
    # Naming the format is the point: "we do not read images" is actionable, "unsupported file" is not.
    assert described.split()[0].lower() in str(exc.value).lower()
    # And it says what WOULD work.
    assert "upload" in str(exc.value).lower()


def test_the_zip_case_specifically_no_longer_reports_a_successful_extraction(tmp_path):
    """RED before SP-01.6a: this exact file produced one Document of container framing plus member
    fragments, passed the empty-extraction guard, and was stored with `status: complete`."""
    path = tmp_path / "archive.zip"
    make_zip(str(path))

    with pytest.raises(UnsupportedDocumentError):
        get_loader("archive.zip", "application/zip", str(path))


# ===========================================================================
# The refusal must NOT become over-eager — this half is the regression guard
# ===========================================================================


def test_a_plain_text_file_with_an_unknown_extension_still_parses(tmp_path):
    # The fallback branch exists for genuinely unknown formats. A text file that merely has an odd
    # extension must keep working exactly as before.
    path = tmp_path / "meeting.notes"
    path.write_text("Escalation owner is the regional lead.", encoding="utf-8")

    loader, known_type, ext = get_loader("meeting.notes", "application/x-unknown", str(path))
    docs = loader.load()

    assert known_type is False  # unchanged: still reported as an unknown type...
    assert "Escalation owner" in " ".join(d.page_content for d in docs)  # ...but still read


@pytest.mark.parametrize(
    "label,text",
    [
        ("japanese", "四半期の売上は前年比で十二パーセント増加しました。"),
        ("arabic", "ارتفعت الإيرادات بنسبة اثني عشر بالمائة على أساس سنوي."),
        ("cyrillic", "Выручка выросла на двенадцать процентов год к году."),
        ("emoji", "Revenue grew 12% 📈 across EMEA 🌍"),
    ],
)
def test_non_latin_text_is_not_mistaken_for_binary(tmp_path, label, text):
    """The sharp regression risk. In UTF-8 every byte of these scripts is >= 0x80, so a
    printable-byte-ratio heuristic calls them binary and refuses a perfectly good document. The check
    decodes instead of counting, precisely so this cannot happen."""
    path = tmp_path / f"{label}.notes"
    path.write_text(text, encoding="utf-8")

    loader, _, _ = get_loader(f"{label}.notes", "application/x-unknown", str(path))
    docs = loader.load()

    assert text[:8] in " ".join(d.page_content for d in docs)


def test_a_known_source_extension_never_reaches_the_check(tmp_path):
    # Files claimed by an earlier branch are untouched by this increment.
    path = tmp_path / "script.py"
    path.write_text("def total(rows):\n    return sum(rows)\n", encoding="utf-8")

    loader, known_type, _ = get_loader("script.py", "text/x-python", str(path))

    assert known_type is True
    assert "def total" in " ".join(d.page_content for d in loader.load())


def test_an_empty_file_is_not_called_unsupported(tmp_path):
    """An empty file is an EMPTY extraction, which the existing 422 guard already reports honestly.
    Relabelling it `unsupported` would be a worse answer, not a better one."""
    path = tmp_path / "blank.notes"
    path.write_bytes(b"")

    loader, _, _ = get_loader("blank.notes", "application/x-unknown", str(path))
    docs = loader.load()
    assert all(not d.page_content.strip() for d in docs)


@pytest.mark.parametrize(
    "sample,expected",
    [
        (b"", False),
        (b"plain ascii text", False),
        ("日本語のテキスト".encode("utf-8"), False),
        (b"has a \x00 nul", True),
        (b"\x89PNG\r\n\x1a\n\x00\x00", True),
    ],
)
def test_looks_like_binary_decodes_rather_than_counts(sample, expected):
    assert looks_like_binary(sample) is expected


# ===========================================================================
# Route level — the verdict reaches the caller, and nothing is stored
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


@pytest.mark.parametrize("filename,mime,build,described", BINARY_CASES)
def test_embed_refuses_an_unsupported_file_with_no_rows(
    client, tmp_path, filename, mime, build, described
):
    path = tmp_path / filename
    build(str(path))

    r = client.post(
        "/embed",
        data={"file_id": f"f-{filename}", "entity_id": "userA"},
        files={"file": (filename, io.BytesIO(path.read_bytes()), mime)},
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["extraction"]["verdict"] == "unsupported"
    assert detail["extraction"]["status"] == "unsupported"
    assert filename in detail["message"]
    assert client.inserted_batches == []


def test_embed_still_accepts_a_real_document(client):
    """Control: the refusal did not swallow the working path."""
    r = client.post(
        "/embed",
        data={"file_id": "f-good", "entity_id": "userA"},
        files={
            "file": (
                "good.txt",
                io.BytesIO(b"Revenue grew twelve percent year over year across EMEA."),
                "text/plain",
            )
        },
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert r.status_code == 200, r.text
    assert sum(len(b) for b in client.inserted_batches) >= 1


# ===========================================================================
# False refusals the independent review found (WPSP1-6A-R MAJOR-1 / MAJOR-2)
#
# Both were regressions introduced by the first version of this guard, and both were silent losses of
# a document that used to ingest. They are the reason the refusal decides by DECODING first and only
# consults magic numbers once the bytes have failed to decode.
# ===========================================================================


@pytest.mark.parametrize(
    "label,encoding",
    [
        ("utf16-bom", "utf-16"),
        ("utf16le-nobom", "utf-16-le"),
        ("utf16be-nobom", "utf-16-be"),
        ("utf32-bom", "utf-32"),
    ],
)
def test_short_bom_less_utf16_is_not_refused(tmp_path, label, encoding):
    """The case chardet alone does NOT cover, and the reason the UTF-16 path exists.

    For a SHORT UTF-16 sample chardet answers `ascii` with confidence 1.00 — and `ascii` is not one
    of the multi-byte encodings the confidence branch accepts, so it contributes nothing. A longer
    sample is detected as `utf-16le` at 0.85 and would pass without the dedicated check, which is
    exactly how this gap hid: the first fixture was long enough to be covered by accident.
    """
    if encoding not in ("utf-16-le", "utf-16-be"):
        pytest.skip("BOM-bearing encodings are settled by the BOM, not by detection")
    path = tmp_path / f"short-{label}.notes"
    path.write_bytes("Hi there.".encode(encoding))

    # Must not raise.
    get_loader(f"short-{label}.notes", "application/octet-stream", str(path))


@pytest.mark.parametrize(
    "label,encoding",
    [
        ("utf16-bom", "utf-16"),
        ("utf16le-nobom", "utf-16-le"),
        ("utf16be-nobom", "utf-16-be"),
        ("utf32-bom", "utf-32"),
    ],
)
def test_utf16_and_utf32_text_is_not_refused(tmp_path, label, encoding):
    """MAJOR-1. UTF-16 encodes ASCII as alternating character/NUL bytes, so the first version's
    "a NUL means binary" rule condemned it — and the docstring asserting no handled encoding emits a
    NUL was simply wrong: `detect_file_encoding` in this same module handles UTF-16 and UTF-32 BOMs.
    A UTF-16 document arriving with an unusual extension and no `text/*` type was refused outright,
    with a message ('not a text-based format') that was itself untrue."""
    text = "Quarterly revenue grew twelve percent across EMEA."
    path = tmp_path / f"{label}.notes"
    path.write_bytes(text.encode(encoding))

    # Not refused — that is what this increment owns.
    loader, _, _ = get_loader(f"{label}.notes", "application/octet-stream", str(path))
    content = " ".join(d.page_content for d in loader.load())

    if encoding in ("utf-16", "utf-32"):
        # With a BOM, TextLoader's autodetect decodes it properly and the text round-trips.
        assert "Quarterly revenue" in content
    else:
        # WITHOUT a BOM, TextLoader's own autodetect still mis-decodes it — a PRE-EXISTING limitation
        # of that loader, not something this guard introduced, and out of scope here. What matters is
        # that the file is no longer REFUSED, and that its characters are present rather than the
        # file being rejected outright.
        assert "Q" in content and "u" in content


@pytest.mark.parametrize(
    "opening",
    [
        "MZ Corporation reported a strong quarter across every region. ",
        "BM means Building Materials in this report, not a bitmap. ",
        "ID3 tags explained, for the audio engineering team. ",
        "GIF87a is an old image format, described here for completeness. ",
    ],
)
def test_prose_beginning_with_a_binary_signature_is_not_refused(tmp_path, opening):
    """MAJOR-2. `MZ` and `BM` are ordinary English bigrams and `ID3`/`GIF87a` occur in technical
    prose, so a two-byte prefix must never outrank the file actually decoding as text. Before the
    correction these were refused AND mis-named — told they were a Windows executable, a BMP image,
    an MP3."""
    path = tmp_path / "note.notes"
    path.write_text(opening * 5, encoding="utf-8")

    loader, _, _ = get_loader("note.notes", "application/octet-stream", str(path))

    assert opening.split()[0] in " ".join(d.page_content for d in loader.load())


def test_a_misnamed_office_file_is_told_what_it_actually_is(tmp_path):
    """MINOR-2. An Office file IS a zip, so a .docx that reaches us with the wrong extension and an
    octet-stream type was refused as 'a ZIP archive' — true, and useless: it sends someone looking
    for an archive they never made."""
    path = tmp_path / "contract.bin"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        z.writestr("word/document.xml", "<w:document/>")

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader("contract.bin", "application/octet-stream", str(path))

    message = str(exc.value).lower()
    assert "office" in message and "wrong file name or type" in message


def test_a_plain_zip_is_still_called_a_zip(tmp_path):
    """Control for the case above: naming Office files must not relabel ordinary archives."""
    path = tmp_path / "archive.zip"
    make_zip(str(path))

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader("archive.zip", "application/zip", str(path))

    assert "zip archive" in str(exc.value).lower()


# ===========================================================================
# The verdict reaches the other routes too (WPSP1-6A-R NOTE-2: only /embed was covered)
# ===========================================================================


def test_text_route_refuses_an_unsupported_file(client, tmp_path):
    path = tmp_path / "photo.png"
    make_png(str(path))

    r = client.post(
        "/text",
        data={"file_id": "f-text-png", "entity_id": "userA"},
        files={"file": ("photo.png", io.BytesIO(path.read_bytes()), "image/png")},
        headers=_hdr(ent=["userA"], act=["read"]),
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["extraction"]["verdict"] == "unsupported"
    assert client.inserted_batches == []


def test_embed_upload_route_refuses_an_unsupported_file(client, tmp_path):
    path = tmp_path / "archive.zip"
    make_zip(str(path))

    r = client.post(
        "/embed-upload",
        data={"file_id": "f-upload-zip", "entity_id": "userA"},
        files={"uploaded_file": ("archive.zip", io.BytesIO(path.read_bytes()), "application/zip")},
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["extraction"]["verdict"] == "unsupported"
    assert client.inserted_batches == []


# ===========================================================================
# OLE2 Office binaries (F-OLE2-NAME)
#
# An OLE2 compound file (legacy .doc/.xls/.ppt, or an ENCRYPTED OOXML file) that arrives with an
# UNKNOWN extension and no Office content type falls through to this branch. It was refused CORRECTLY
# — verdict `unsupported`, zero rows — but named "not a text-based format", the one string the
# module's own comment says is NOT actionable, because `_BINARY_SIGNATURES` had no OLE2 entry and so
# `describe_unsupported_binary` returned None. Measured on origin/main through /embed with a genuine
# OLE2 container: 422, verdict `unsupported`, units_total=0, message ending "is not a text-based
# format". This section names it as an Office binary WITHOUT making the refusal over-eager, and the
# controls prove the .xls / .xlsx / .doc branches — which route by extension or content type BEFORE
# this table is ever consulted — keep their existing verdicts.
# ===========================================================================


_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_XLS_MIME = "application/vnd.ms-excel"


def make_ole2_office_binary(path, tmp_path):
    """A GENUINE OLE2 compound file, built at test time: a real encrypted OOXML workbook produced by
    msoffcrypto's own encryptor. Encryption wraps the package in an OLE2 container, so its first bytes
    ARE the CFB magic — exactly the shape a legacy .doc/.xls/.ppt or any encrypted Office file has, and
    the shape that used to be named generically. Not a hand-rolled header: a header with random bytes
    behind it could pass the weaker UTF-16/chardet text checks by accident, so the fixture must be a
    real high-entropy container to exhibit the defect through the whole decision."""
    from openpyxl import Workbook
    from msoffcrypto.format.ooxml import OOXMLFile

    plain = tmp_path / "plain-source.xlsx"
    wb = Workbook()
    wb.active.title = "S1"
    wb.active["A1"] = "confidential total"
    wb.save(str(plain))
    with open(plain, "rb") as src, open(path, "wb") as out:
        OOXMLFile(src).encrypt("hunter2", out)


def test_ole2_head_is_named_as_an_office_binary_not_none(tmp_path):
    """Unit level: the root cause was `describe_unsupported_binary` returning None for OLE2. It now
    names the format, and `looks_like_binary` agrees the bytes are binary (they are)."""
    path = tmp_path / "container.bin"
    make_ole2_office_binary(path, tmp_path)
    head = path.read_bytes()[:8192]

    assert head.startswith(_OLE2_MAGIC)  # fixture premise: it really is an OLE2 container
    described = describe_unsupported_binary(head)
    assert described is not None
    assert "office" in described.lower()
    assert looks_like_binary(head) is True


def test_ole2_with_unknown_extension_is_named_as_office_not_generic(tmp_path):
    """The finding. A genuine OLE2 file with an unknown extension is still REFUSED (`unsupported`,
    the refusal must not become over-eager the other way), but is now NAMED as an Office binary
    instead of the un-actionable "not a text-based format"."""
    path = tmp_path / "container.bin"
    make_ole2_office_binary(path, tmp_path)

    with pytest.raises(UnsupportedDocumentError) as exc:
        get_loader("container.bin", "application/octet-stream", str(path))

    message = str(exc.value)
    assert exc.value.verdict == "unsupported"
    # Named, not generic — this is the exact string the fix removes for OLE2.
    assert "not a text-based format" not in message.lower()
    assert "office" in message.lower()
    # Still actionable about what WOULD work.
    assert "upload" in message.lower()


def test_embed_refuses_ole2_unknown_extension_named_with_no_rows(client, tmp_path):
    """Route level, the wire the caller sees. Same 422 / `unsupported` / zero rows as before — the
    refusal is unchanged — but the message on the wire now names the Office binary."""
    path = tmp_path / "container.bin"
    make_ole2_office_binary(path, tmp_path)

    r = client.post(
        "/embed",
        data={"file_id": "f-ole2-unknown", "entity_id": "userA"},
        files={"file": ("container.bin", io.BytesIO(path.read_bytes()), "application/octet-stream")},
        headers=_hdr(ent=["userA"], act=["write"]),
    )

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["extraction"]["verdict"] == "unsupported"
    assert detail["extraction"]["status"] == "unsupported"
    assert detail["extraction"]["units_total"] == 0
    assert "office" in detail["message"].lower()
    assert "not a text-based format" not in detail["message"].lower()
    assert client.inserted_batches == []  # zero rows written for a refused file


# --- controls: the .xls / .xlsx / .doc branches are untouched ---------------


def test_encrypted_xlsx_keeps_its_encrypted_verdict(tmp_path):
    """CONTROL. An encrypted .xlsx IS an OLE2 container, but it arrives with its real extension/type
    and is routed to SheetExcelLoader BEFORE the signature table. It must still get `encrypted`, not
    be relabelled `unsupported` by the new OLE2 signature."""
    path = tmp_path / "protected.xlsx"
    make_ole2_office_binary(path, tmp_path)  # a genuine encrypted OOXML workbook

    loader, known_type, ext = get_loader("protected.xlsx", _XLSX_MIME, str(path))
    assert isinstance(loader, SheetExcelLoader) and known_type is True and ext == "xlsx"

    with pytest.raises(EncryptedDocumentError) as exc:
        loader.load()
    assert exc.value.verdict == "encrypted"
    assert not isinstance(exc.value, UnsupportedDocumentError)


def test_genuine_legacy_xls_still_parses_and_is_not_refused():
    """CONTROL. A REAL legacy .xls (OLE2), the case the module cannot generate. It routes to
    SheetExcelLoader by extension, parses, and keeps its sheet-level citation — the OLE2 signature
    entry does not steal it into an `unsupported` refusal. `describe_unsupported_binary` WOULD name
    its head as Office, which is exactly why the routing-before-signatures order matters and is
    asserted here: the file must never reach that function."""
    loader, known_type, ext = get_loader("legacy.xls", _XLS_MIME, _LEGACY_XLS)
    assert isinstance(loader, SheetExcelLoader) and known_type is True and ext == "xls"

    docs = loader.load()
    text = " ".join(d.page_content for d in docs)
    assert "LEGACY-CONTENT-MARKER-4471" in text  # it genuinely parsed, not refused
    assert any(d.metadata.get("page_name") == "Sheet1" for d in docs)  # citation preserved

    # The fixture head DOES match the new signature; the .xls branch owning it first is the invariant.
    with open(_LEGACY_XLS, "rb") as f:
        assert describe_unsupported_binary(f.read(8192)) is not None


def test_doc_branch_is_unchanged_by_the_ole2_signature(tmp_path):
    """CONTROL. A .doc arrives on the Word branch (Docx2txtLoader) by extension/content type, never
    through the fallback signature check. Routing it must not raise the OLE2 `unsupported` verdict —
    whatever the Word loader then does with the bytes is out of this row's scope (see F-LEGACY3)."""
    path = tmp_path / "memo.doc"
    make_ole2_office_binary(path, tmp_path)

    # get_loader routes without reading bytes for a signature; it must not raise here.
    loader, known_type, ext = get_loader("memo.doc", "application/msword", str(path))
    assert type(loader).__name__ == "Docx2txtLoader"
    assert ext == "doc"


def test_ole2_signature_and_excel_container_check_share_one_literal():
    """CONTROL for the de-duplication: `SheetExcelLoader`'s container check and the fallback signature
    table must use the SAME OLE2 literal, or an encrypted-vs-corrupt decision could drift from the
    naming decision. Pins them to one module-level constant."""
    assert _OLE2_MAGIC == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    assert SheetExcelLoader._OLE2_MAGIC is _OLE2_MAGIC

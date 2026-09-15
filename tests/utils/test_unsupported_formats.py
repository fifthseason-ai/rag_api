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
    UnsupportedDocumentError,
    get_loader,
    looks_like_binary,
)

_SECRET = "testsecret"


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
    assert loader.load() == [] or all(not d.page_content.strip() for d in loader.load())


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
                           document_origin_type=None, subscription_id=None, executor=None):
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

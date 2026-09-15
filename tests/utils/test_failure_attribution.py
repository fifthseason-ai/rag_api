"""Never blame the uploader's file for this service's own faults (KI-02 SP-01.10).

SP-01.5 fixed the production Excel failure by pinning `msoffcrypto-tool`. It fixed the INSTANCE. The
CLASS was untouched, and the class is the interesting part:

    ModuleNotFoundError("No module named 'msoffcrypto'")
        -> 400 "Error during file processing: No module named 'msoffcrypto'"

That is the production failure verbatim. A dependency missing from OUR image was reported to the
uploader as a problem with THEIR file, over a status code whose whole meaning is "your request is bad,
do not retry" — so Core's listener recorded the file as failed rather than retrying it after the image
was fixed. Every future missing dependency does the same thing. Reproduced this session against the real
seam, along with two siblings:

    MemoryError("cannot allocate 4.2 GiB")   -> 400  (a server fault sold as a client fault)
    ValueError("/tmp/uploads/abc123/q4-forecast.xlsx: corrupt sector table at 0x1f40")
                                             -> 400  echoing our internal temp path back to the caller

`str(e)` is handed to the caller raw at `document_routes.py:1519/1527` and `:1662`.

The correction is an ATTRIBUTION, not a new verdict vocabulary:

  * a recognised SERVICE fault  -> 503, explicitly not a statement about the file, safe to retry;
  * anything else               -> the status is UNCHANGED (400), but the message loses the raw
    exception and says plainly that the cause is not established;
  * a terminal FILE verdict     -> 422, exactly as SP-01.5 left it. Untouched.

`detail` stays a plain STRING on the new paths ON PURPOSE. Core's direct-upload consumer interpolates
`detail` into the user's toast (`crud.js:341`), so an object there renders as `[object Object]` — a live
defect this lane found and recorded separately. The machine-readable signal is the STATUS CODE, which
every consumer already reads, and the human signal is the sentence. Nothing has to be coordinated for
both to be correct.

Every fault below is INJECTED at the loader; no real dependency is removed and no real memory exhausted.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from unittest.mock import patch

from app.routes.document_routes import load_file_content
from app.utils.document_loader import EncryptedDocumentError


MARKER = "INTERNALS-THAT-MUST-NOT-REACH-THE-CALLER-7f3a"
TEMP_PATH = "/tmp/uploads/abc123/q4-forecast.xlsx"


class RaisingLoader:
    """A loader whose `lazy_load` fails the way a real one would."""

    def __init__(self, exc):
        self._exc = exc

    def lazy_load(self):
        raise self._exc


class WorkingLoader:
    def lazy_load(self):
        from langchain_core.documents import Document

        return [Document(page_content="real content", metadata={})]


def drive(exc_or_loader, filename="q4-forecast.xlsx"):
    """Run the REAL seam with the loader replaced. Returns the raised HTTPException."""
    loader = (
        exc_or_loader
        if hasattr(exc_or_loader, "lazy_load")
        else RaisingLoader(exc_or_loader)
    )
    executor = ThreadPoolExecutor(max_workers=1)
    with patch(
        "app.routes.document_routes.get_loader",
        return_value=(loader, True, "xlsx"),
    ):
        return asyncio.run(
            load_file_content(
                filename, "application/vnd.ms-excel", TEMP_PATH, executor
            )
        )


def detail_text(exc: HTTPException) -> str:
    """The caller-visible sentence, whatever shape `detail` takes."""
    d = exc.detail
    return d if isinstance(d, str) else str(d.get("message", d))


# ── SERVICE faults: our problem, never reported as theirs ─────────────────────────────────────────


@pytest.mark.parametrize(
    "exc,label",
    [
        (ModuleNotFoundError(f"No module named '{MARKER}'"), "missing dependency"),
        (ImportError(f"cannot import name {MARKER}"), "broken import"),
        (MemoryError(f"cannot allocate 4.2 GiB {MARKER}"), "out of memory"),
        (OSError(f"[Errno 28] No space left on device: {MARKER}"), "disk full"),
        (PermissionError(f"[Errno 13] Permission denied: {MARKER}"), "permission"),
        (FileNotFoundError(f"[Errno 2] No such file: {MARKER}"), "our temp file vanished"),
        (ConnectionError(f"connection reset {MARKER}"), "connection"),
        (TimeoutError(f"timed out {MARKER}"), "timeout"),
        (RecursionError(f"maximum recursion depth exceeded {MARKER}"), "recursion"),
    ],
)
def test_a_service_fault_is_never_reported_as_a_bad_file(exc, label):
    """RED-first. Every one of these returned 400 — "your file is bad" — before SP-01.10."""
    with pytest.raises(HTTPException) as caught:
        drive(exc)
    err = caught.value

    # 5xx: this is us. 4xx would tell the caller their request was wrong and not to retry it.
    assert err.status_code == 503, f"{label} must not be a client error"

    text = detail_text(err)
    # The sentence must actively exonerate the file, not merely fail to accuse it.
    assert "not with your file" in text, f"{label}: {text}"
    # Retryability is the operational half — the production Excel files were marked failed, not retried.
    assert "try again" in text.lower(), f"{label}: {text}"
    # Our internals never travel to the caller.
    assert MARKER not in text, f"{label} leaked the raw exception: {text}"
    assert TEMP_PATH not in text, f"{label} leaked our temp path: {text}"


def test_the_production_excel_failure_specifically():
    """The exact exception that took Excel down in production, asserted by name.

    Pinning msoffcrypto-tool fixed this ONE dependency. This asserts the SHAPE is handled, so the next
    missing dependency is an honest 503 instead of an accusation.
    """
    with pytest.raises(HTTPException) as caught:
        drive(ModuleNotFoundError("No module named 'msoffcrypto'"))
    err = caught.value
    assert err.status_code == 503
    text = detail_text(err)
    assert "msoffcrypto" not in text, "the dependency name is our business, not the uploader's"
    assert "q4-forecast.xlsx" in text, "the caller must still know WHICH file"


# ── UNDETERMINED: honest about not knowing, and still not leaking ─────────────────────────────────


def test_an_unclassified_failure_keeps_its_status_but_loses_the_internals():
    """RED-first on the leak. The status stays 400 deliberately: changing it would change retry
    behaviour for every consumer, and there is no evidence to support a different code. What changes is
    that the caller stops receiving our exception text."""
    with pytest.raises(HTTPException) as caught:
        drive(ValueError(f"{TEMP_PATH}: corrupt sector table at 0x1f40 {MARKER}"))
    err = caught.value

    assert err.status_code == 400, "status deliberately unchanged"
    text = detail_text(err)
    assert MARKER not in text
    assert TEMP_PATH not in text
    assert "0x1f40" not in text
    # Honest about the limit of what we know — it does NOT assert the file is at fault.
    assert "not established" in text
    assert "q4-forecast.xlsx" in text


def test_every_failure_carries_a_reference_that_ties_it_to_the_log():
    """Withholding internals is only acceptable if an operator can still find them."""
    refs = []
    for exc in (ModuleNotFoundError("dep"), ValueError("parse")):
        with pytest.raises(HTTPException) as caught:
            drive(exc)
        text = detail_text(caught.value)
        assert "Reference:" in text, text
        ref = text.split("Reference:")[1].strip().rstrip(".")
        assert len(ref) >= 8, f"reference too short to be useful: {ref!r}"
        refs.append(ref)
    assert refs[0] != refs[1], "a shared reference would point an operator at the wrong failure"


def test_the_reference_actually_appears_in_the_server_log(caplog):
    """Test-the-test for the line above: a reference nobody logged is decoration."""
    import logging

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as caught:
            drive(ValueError(f"something internal {MARKER}"))
    text = detail_text(caught.value)
    ref = text.split("Reference:")[1].strip().rstrip(".")
    logged = caplog.text
    assert ref in logged, "the reference handed to the caller is not in the log"
    # And the detail the CALLER was denied must be present for the OPERATOR.
    assert MARKER in logged, "the internals were withheld from the log too, which helps nobody"


# ── CONTROLS: the SP-01.5 verdict path and the success path are untouched ─────────────────────────


def test_a_terminal_file_verdict_is_unchanged():
    """SP-01.5's 422 is a statement about the FILE and is correct. It must not be swept into the new
    attribution logic — an encrypted workbook IS the uploader's to fix."""
    with pytest.raises(HTTPException) as caught:
        drive(EncryptedDocumentError("this workbook is password-protected"))
    err = caught.value
    assert err.status_code == 422
    assert isinstance(err.detail, dict), "the structured verdict shape is part of the contract"
    assert err.detail["extraction"]["verdict"] == "encrypted"
    assert "password-protected" in err.detail["message"]
    # It must NOT have acquired the service-fault language.
    assert "not with your file" not in err.detail["message"]


def test_a_file_that_loads_still_loads():
    """The cheapest possible regression: the happy path is not routed through any of this."""
    data, known_type, file_ext = drive(WorkingLoader())
    assert len(data) == 1
    assert data[0].page_content == "real content"
    assert known_type is True
    assert file_ext == "xlsx"


def test_a_service_fault_is_distinguishable_from_a_file_verdict_by_status_alone():
    """The machine-readable half. A consumer must be able to tell "retry later" from "this file will
    never work" without parsing prose — that is what the status code is for, and it is the reason the
    new paths keep `detail` a plain string instead of inventing a second vocabulary."""
    statuses = {}
    for label, exc in (
        ("service", MemoryError("oom")),
        ("file", EncryptedDocumentError("locked")),
        ("undetermined", ValueError("no idea")),
    ):
        with pytest.raises(HTTPException) as caught:
            drive(exc)
        statuses[label] = caught.value.status_code
    assert statuses["service"] == 503
    assert statuses["file"] == 422
    assert statuses["undetermined"] == 400
    assert len(set(statuses.values())) == 3, "the three cases must not collapse into one answer"


def test_a_deliberate_http_answer_from_the_loader_is_never_re_wrapped():
    """`get_loader` can raise its own HTTPException — an answer someone already chose deliberately.
    Re-attributing it would overwrite a considered status and message with a generic one."""
    deliberate = HTTPException(status_code=413, detail="That file is larger than this service accepts.")
    executor = ThreadPoolExecutor(max_workers=1)
    with patch("app.routes.document_routes.get_loader", side_effect=deliberate):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                load_file_content("big.xlsx", "application/vnd.ms-excel", TEMP_PATH, executor)
            )
    assert caught.value.status_code == 413
    assert caught.value.detail == "That file is larger than this service accepts."


# ── SP-01.10b — the faults independent review found still misattributed ───────────────────────────
#
# The first cut recognised a TYPED set: ImportError, MemoryError, OSError, RecursionError,
# asyncio.TimeoutError. Review checked the hierarchy in-container and found the set both too NARROW and
# too WIDE, and an inline comment that asserted something false.
#
# TOO NARROW, and this is the part that matters: `psycopg2.OperationalError`,
# `sqlalchemy.exc.OperationalError`, `httpx.ConnectError`, `redis.ConnectionError`, botocore's
# connection errors and `concurrent.futures.TimeoutError` are NOT OSError subclasses (verified, not
# assumed). A vector-DB or embeddings-API outage therefore landed on the non-retryable 400 path -- which
# is the EXACT production harm this increment exists to close, left open for the most likely cause of it.
# The comment on the embed handler claimed "a connection error to the vector DB ... must not be returned
# as 'your file is bad'" while the code did precisely that.
#
# TOO WIDE: `PIL.UnidentifiedImageError` IS an OSError, and it is a statement about CONTENT. Excusing it
# as a retryable service fault would retry a file that can never work.
#
# Detection is by MODULE for the infrastructure libraries: this file must not import a driver it may not
# have, and a driver upgrade that renames a class should not silently reopen the hole.


REAL_INFRA_FAULTS = [
    ("psycopg2", "OperationalError", ("connection to server failed",)),
    ("sqlalchemy.exc", "OperationalError", ("SELECT 1", None, None)),
    ("httpx", "ConnectError", ("connection refused",)),
    ("redis.exceptions", "ConnectionError", ("connection lost",)),
]


@pytest.mark.parametrize("module_name,cls_name,args", REAL_INFRA_FAULTS)
def test_a_real_infrastructure_error_is_a_service_fault(module_name, cls_name, args):
    """RED-first, against the REAL exception classes rather than a stand-in."""
    module = pytest.importorskip(module_name)
    exc_cls = getattr(module, cls_name)
    assert not issubclass(exc_cls, OSError), (
        f"{module_name}.{cls_name} is now an OSError; this test no longer proves anything"
    )
    with pytest.raises(HTTPException) as caught:
        drive(exc_cls(*args))
    assert caught.value.status_code == 503, f"{module_name}.{cls_name} must be retryable"
    assert "not with your file" in detail_text(caught.value)


def test_concurrent_futures_timeout_is_covered_too():
    """`asyncio.TimeoutError is concurrent.futures.TimeoutError` is FALSE on this Python, so listing one
    does not cover the other. Verified in-container rather than assumed."""
    import asyncio as _asyncio
    import concurrent.futures as _futures

    assert _asyncio.TimeoutError is not _futures.TimeoutError
    with pytest.raises(HTTPException) as caught:
        drive(_futures.TimeoutError("the pool timed out"))
    assert caught.value.status_code == 503


def test_detection_is_by_module_so_an_unimported_driver_still_counts():
    """The mechanism, isolated from whatever happens to be installed: an exception whose module is an
    infrastructure package is ours even though this file never imports that package."""

    class SomeFutureDriverError(Exception):
        pass

    SomeFutureDriverError.__module__ = "psycopg.errors"
    with pytest.raises(HTTPException) as caught:
        drive(SomeFutureDriverError("server closed the connection unexpectedly"))
    assert caught.value.status_code == 503


def test_a_content_error_that_happens_to_be_an_oserror_is_NOT_excused():
    """RED-first the other way. `PIL.UnidentifiedImageError` inherits OSError but says the BYTES are
    unreadable. A retryable 503 here would retry a file that can never work."""
    PIL = pytest.importorskip("PIL")
    from PIL import UnidentifiedImageError

    assert issubclass(UnidentifiedImageError, OSError), "the premise of this test has changed"
    with pytest.raises(HTTPException) as caught:
        drive(UnidentifiedImageError("cannot identify image file"))
    err = caught.value
    assert err.status_code == 400, "a content fault must not be sold as retryable"
    assert "not with your file" not in detail_text(err)
    assert "not established" in detail_text(err)


def test_a_document_parser_error_is_never_excused_as_infrastructure():
    """The deny-list must not be so broad that a parser's own failure looks like an outage."""

    class ParseFailure(Exception):
        pass

    for parser_module in ("openpyxl.reader.excel", "pypdf.errors", "unstructured.partition"):
        ParseFailure.__module__ = parser_module
        with pytest.raises(HTTPException) as caught:
            drive(ParseFailure("bad record"))
        assert caught.value.status_code == 400, f"{parser_module} must not read as infrastructure"

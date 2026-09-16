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
    """A pool timeout is ours, on every Python this can run on.

    The first version of this test asserted `asyncio.TimeoutError is not concurrent.futures.TimeoutError`.
    That is TRUE on 3.10 -- which is what the Dockerfile ships -- and FALSE from 3.11, where both names
    are the builtin `TimeoutError`, so CI (3.12) reds it. The identity was never the point: it was my
    reason for listing `asyncio.TimeoutError` separately, and reasons do not belong in assertions.

    What must hold on both is the BEHAVIOUR, so that is what is asserted. On 3.10 the executor converts
    the pool timeout to `asyncio.TimeoutError` as it crosses `run_in_executor`; from 3.11 the two names
    are one class that also subclasses OSError. Different routes, same answer.
    """
    import concurrent.futures as _futures

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


# ── SP-01.10c — the module-root set was too broad, and blind to wrapping ──────────────────────────
#
# Re-review found a REGRESSION that 10b introduced. A DB driver raises BOTH kinds of error: an outage
# (`OperationalError` -- ours, transient, retryable) and a complaint about the VALUE being written
# (`DataError` -- permanent, and in this service that value is text extracted from the uploaded file: a
# NUL byte, an invalid UTF-8 sequence, an over-length field). Module-root detection cannot tell them
# apart, so 10b promoted `DataError` to a retryable 503 and would retry, forever, a file that can never
# work. That is precisely the bug-shape the PIL deny-list exists to prevent, arriving through psycopg2
# instead of PIL -- and 9228ced had it right by accident, because DataError is not an OSError.
#
# The second gap: langchain wraps provider calls in tenacity, and the classifier read only the TOP
# exception's module. A real outage surfacing as `tenacity.RetryError` (cause: httpx/botocore) read as
# undetermined. The cause chain is now walked, content faults first so a wrapper can never launder one.


def test_a_db_data_error_is_the_files_fault_not_an_outage():
    """RED-first. `DataError` is about the VALUE, which here is text extracted from the upload."""
    psycopg2 = pytest.importorskip("psycopg2")
    with pytest.raises(HTTPException) as caught:
        drive(psycopg2.DataError("invalid byte sequence for encoding UTF8: 0x00"))
    assert caught.value.status_code == 400, "a permanent content fault must not be sold as retryable"
    assert "not with your file" not in detail_text(caught.value)


def test_sqlalchemy_data_and_integrity_errors_are_not_outages():
    sa = pytest.importorskip("sqlalchemy.exc")
    for cls_name in ("DataError", "IntegrityError"):
        exc_cls = getattr(sa, cls_name)
        with pytest.raises(HTTPException) as caught:
            drive(exc_cls("INSERT ...", None, Exception("value too long")))
        assert caught.value.status_code == 400, f"{cls_name} must not be retryable"


def test_the_outage_errors_from_the_same_driver_are_still_service_faults():
    """CONTROL. Narrowing must not undo 10b: the SAME libraries still report outages as ours."""
    psycopg2 = pytest.importorskip("psycopg2")
    sa = pytest.importorskip("sqlalchemy.exc")
    for exc in (
        psycopg2.OperationalError("could not connect to server"),
        psycopg2.InterfaceError("connection already closed"),
        sa.OperationalError("SELECT 1", None, None),
    ):
        with pytest.raises(HTTPException) as caught:
            drive(exc)
        assert caught.value.status_code == 503, f"{type(exc).__name__} is still an outage"


def test_a_wrapped_outage_is_recognised_through_the_cause_chain():
    """RED-first. langchain wraps provider calls in tenacity; reading only the top exception's module
    meant a real outage arrived as undetermined."""
    httpx = pytest.importorskip("httpx")
    inner = httpx.ConnectError("connection refused")
    outer = RuntimeError("retries exhausted")
    outer.__cause__ = inner
    with pytest.raises(HTTPException) as caught:
        drive(outer)
    assert caught.value.status_code == 503
    assert "not with your file" in detail_text(caught.value)


def test_a_wrapper_can_never_launder_a_content_fault_into_an_outage():
    """The dangerous direction of walking the chain: if ANY link is a content fault, the whole thing is.
    Otherwise wrapping a DataError in a retry would make a permanent failure look transient."""
    psycopg2 = pytest.importorskip("psycopg2")
    inner = psycopg2.DataError("invalid byte sequence 0x00")
    outer = RuntimeError("retries exhausted")
    outer.__cause__ = inner
    # And the reverse nesting, so neither order launders it.
    other = psycopg2.OperationalError("connection lost")
    other.__context__ = psycopg2.DataError("invalid byte sequence 0x00")
    for exc in (outer, other):
        with pytest.raises(HTTPException) as caught:
            drive(exc)
        assert caught.value.status_code == 400, "a content fault anywhere in the chain wins"


def test_the_cause_chain_walk_is_bounded_and_survives_a_cycle():
    """A self-referential chain must not hang the request."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    with pytest.raises(HTTPException) as caught:
        drive(a)
    assert caught.value.status_code == 400


def test_the_module_denylist_covers_PIL_errors_beyond_the_named_ones():
    """My own mutation run caught this: emptying `_CONTENT_FAULT_MODULE_ROOTS` left every test green,
    because `UnidentifiedImageError` is ALSO on the class-name list. The module entry exists for PIL's
    OTHER content errors -- a decompression bomb, a bad palette -- which inherit OSError and are just as
    permanent. Exercised with a synthetic class so the test does not depend on which exceptions this
    version of Pillow happens to define."""

    class SomeOtherPillowError(OSError):
        pass

    SomeOtherPillowError.__module__ = "PIL.Image"
    assert SomeOtherPillowError.__name__ not in ("UnidentifiedImageError", "DataError")
    with pytest.raises(HTTPException) as caught:
        drive(SomeOtherPillowError("broken data stream when reading image file"))
    assert caught.value.status_code == 400, "an OSError from PIL is about the bytes, not our storage"
    assert "not with your file" not in detail_text(caught.value)


def test_a_suppressed_context_is_not_treated_as_a_cause():
    """RED-first, from round-3 review. `raise X from None` sets `__suppress_context__`: the developer has
    said explicitly that whatever was being handled is INCIDENTAL. Python's own traceback machinery hides
    it, and the classifier must mirror that — otherwise a genuine transient outage raised while some
    unrelated content error happened to be in flight gets marked permanent, which is the laundering
    problem running in reverse and costs a file that would have worked."""
    httpx = pytest.importorskip("httpx")
    psycopg2 = pytest.importorskip("psycopg2")

    try:
        try:
            raise psycopg2.DataError("invalid byte sequence 0x00")
        except psycopg2.DataError:
            raise httpx.ConnectError("connection refused") from None
    except httpx.ConnectError as outage:
        captured = outage

    assert captured.__suppress_context__ is True
    assert isinstance(captured.__context__, psycopg2.DataError), "the premise of this test has changed"

    with pytest.raises(HTTPException) as caught:
        drive(captured)
    assert caught.value.status_code == 503, "a suppressed context must not make an outage permanent"
    assert "not with your file" in detail_text(caught.value)


def test_an_UNSUPPRESSED_context_still_counts():
    """CONTROL. Only the explicit signal is honoured; an ordinary implicit context is still read, so the
    fix cannot be used to hide a real content fault."""
    httpx = pytest.importorskip("httpx")
    psycopg2 = pytest.importorskip("psycopg2")

    try:
        try:
            raise psycopg2.DataError("invalid byte sequence 0x00")
        except psycopg2.DataError:
            raise httpx.ConnectError("connection refused")
    except httpx.ConnectError as outage:
        captured = outage

    assert captured.__suppress_context__ is False
    with pytest.raises(HTTPException) as caught:
        drive(captured)
    assert caught.value.status_code == 400, "an unsuppressed content context still wins"


def test_a_suppressed_context_does_not_hide_a_content_fault_in_the_CAUSE():
    """`from None` suppresses the CONTEXT only. An explicit `__cause__` is never suppressed, so a content
    fault deliberately chained still wins."""
    psycopg2 = pytest.importorskip("psycopg2")
    httpx = pytest.importorskip("httpx")

    outage = httpx.ConnectError("connection refused")
    outage.__cause__ = psycopg2.DataError("invalid byte sequence 0x00")
    outage.__suppress_context__ = True
    with pytest.raises(HTTPException) as caught:
        drive(outage)
    assert caught.value.status_code == 400


# ── SP-01.13 — the residual caller-facing str(e) sites on this lane's own upload/extract path ─────
#
# SP-01.10 fixed the embed handlers and named these three as scoped out. They echo str(e) to the caller:
# a disk or permission failure while saving to OUR temp directory puts our temp path in the caller's
# message. Routed through the same describe_failure -- a save failure is our storage (503); the /text
# else-branch keeps its status but loses str(e). The pandoc special-case is preserved: it is honest and
# actionable and must not be swept into the generic path.

import app.routes.document_routes as dr


import tempfile, os as _os
_TMPDIR = tempfile.mkdtemp(prefix="ki02-sp0113-")
TEMP = _os.path.join(_TMPDIR, "q4-forecast.xlsx")
INTERNAL = "PERMISSION-DENIED-INTERNAL-DETAIL-9x7"


class _FakeUpload:
    def __init__(self, filename):
        self.filename = filename

    async def read(self, _n):
        raise PermissionError(f"[Errno 13] Permission denied: '{TEMP}' {INTERNAL}")

    @property
    def file(self):
        raise PermissionError(f"[Errno 13] Permission denied: '{TEMP}' {INTERNAL}")


def _detail(exc):
    d = exc.detail
    return d if isinstance(d, str) else str(d.get("message", d))


def test_async_save_failure_does_not_echo_our_path_or_exception():
    import asyncio

    up = _FakeUpload("q4-forecast.xlsx")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(dr.save_upload_file_async(up, TEMP))
    err = caught.value
    # A save failure is our storage: ours, transient, retryable.
    assert err.status_code == 503
    text = _detail(err)
    assert TEMP not in text
    assert INTERNAL not in text
    assert "Errno 13" not in text
    assert "not with your file" in text  # a save failure is never the uploaded file's fault
    assert "q4-forecast.xlsx" in text  # the caller still learns WHICH file


def test_sync_save_failure_does_not_echo_our_path_or_exception():
    up = _FakeUpload("q4-forecast.xlsx")
    with pytest.raises(HTTPException) as caught:
        dr.save_upload_file_sync(up, TEMP)
    err = caught.value
    assert err.status_code == 503
    text = _detail(err)
    assert TEMP not in text
    assert INTERNAL not in text
    assert "not with your file" in text


def test_save_failure_carries_a_reference_to_the_log(caplog):
    import asyncio
    import logging

    up = _FakeUpload("q4-forecast.xlsx")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(dr.save_upload_file_async(up, TEMP))
    text = _detail(caught.value)
    assert "Reference:" in text
    ref = text.split("Reference:")[1].strip().rstrip(".")
    assert ref in caplog.text  # the operator can find the withheld detail
    assert INTERNAL in caplog.text  # and the detail IS in the log, not discarded

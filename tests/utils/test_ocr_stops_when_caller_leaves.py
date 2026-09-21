"""A caller who leaves must stop the OCR too, not only the write (FILES-01 F04B).

WHY THIS FILE EXISTS. Measured 2026-09-21 on a real uvicorn server with a real 12-page
scanned PDF: the client gave up at 2.0 s and all 12 pages were still OCR'd, the last at
17.6 s -- about 15 s of CPU for nobody (F04 already guaranteed nothing would be STORED).
The loader has had a stop flag checked between pages all along, but it was set only when
the handler was cancelled, and Starlette never cancels a handler for a disconnect.

Now the request's disconnect probe trips that flag. Page 1's OCR is gated on an event the
test releases only after the client has given up, so "the caller left during OCR" is
arranged, not hoped for.
"""

import asyncio
import datetime
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import jwt
import pytest
import uvicorn

from main import app
from app.routes import document_routes
from app.utils import ocr as ocr_module
from tests.utils.test_replace_not_accumulate import FakeStore
from tests.utils.test_scanned_pdf_ocr import INVOICE, scanned_pdf

_SECRET = "testsecret"
PAGES = 4


@pytest.fixture()
def server(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=4)

    gate = threading.Event()
    first_page_started = threading.Event()
    completed = []
    real_ocr_page = ocr_module.ocr_page

    def gated_ocr_page(page, budget):
        if not first_page_started.is_set():
            first_page_started.set()
            assert gate.wait(30), "test never released the first page"
        result = real_ocr_page(page, budget)
        completed.append(result)
        return result

    # The loader imports ocr_page from the module at call time, so this reaches it.
    monkeypatch.setattr(ocr_module, "ocr_page", gated_ocr_page)

    # Record the request's disconnect probe with its server loop, so the test can ask the
    # server whether it has seen the caller leave (a fixed sleep is not a synchronisation).
    probes = []
    real_probe_factory = document_routes.caller_still_waiting

    def recording_probe_factory(request):
        probe = real_probe_factory(request)
        probes.append((probe, asyncio.get_running_loop()))
        return probe

    monkeypatch.setattr(document_routes, "caller_still_waiting", recording_probe_factory)

    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not srv.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    base = f"http://127.0.0.1:{srv.servers[0].sockets[0].getsockname()[1]}"
    try:
        yield base, store, gate, first_page_started, completed, probes
    finally:
        gate.set()
        srv.should_exit = True
        thread.join(15)


def _post(base, timeout):
    os.environ["JWT_SECRET"] = _SECRET
    tok = jwt.encode(
        {"id": "u", "tid": "tenantA", "ent": ["userA"], "act": ["write"],
         "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
        _SECRET, algorithm="HS256",
    )
    pdf = scanned_pdf(INVOICE, pages=PAGES, w=850, h=1100, size=21)
    return httpx.post(
        f"{base}/embed", timeout=timeout,
        data={"file_id": "scan", "entity_id": "userA"},
        files={"file": ("scan.pdf", io.BytesIO(pdf), "application/pdf")},
        headers={"Authorization": f"Bearer {tok}"},
    )


def _wait_for(predicate, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_ocr_stops_after_the_page_in_flight_once_the_caller_has_gone(server):
    base, store, gate, first_page_started, completed, probes = server
    with pytest.raises(httpx.ReadTimeout):
        _post(base, timeout=1.0)
    assert first_page_started.is_set(), "the request never reached OCR"
    assert _wait_for(
        lambda: any(
            asyncio.run_coroutine_threadsafe(p(), loop).result(5) is False
            for p, loop in list(probes)
        ),
        10,
    ), "the server never noticed the disconnect"
    time.sleep(0.4)  # one poll period (0.25 s) of the extraction stop-watcher, plus margin
    gate.set()

    # Give the remaining pages every chance to run if the stop did not work.
    _wait_for(lambda: len(completed) >= PAGES, 10)
    # At most the page already in flight: ocr_page checks the stop flag before it starts,
    # so when the flag lands first even that page is skipped (measured: 0).
    assert len(completed) <= 1, (
        f"{len(completed)} of {PAGES} pages were OCR'd after the caller had left"
    )
    assert store.rows == [], "a departed caller's scan was stored"


def test_a_caller_who_waits_gets_every_page(server):
    base, store, gate, first_page_started, completed, probes = server
    gate.set()
    r = _post(base, timeout=120)
    assert r.status_code == 200, r.text
    assert len(completed) == PAGES
    assert store.rows, "a waiting caller's scan was not stored"

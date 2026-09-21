"""The summarizer worker is bounded so a hung model call cannot hold a thread (FILES-01 F04).

`summarize_files` runs a synchronous LLM call inside a thread-pool worker (max 8 threads).
Unbounded, one slow/hung model call holds a worker and its request indefinitely; enough of
them wedge the service. F04 requires reachable timeout/resource protection on the worker
path. `_summarize_within_timeout` bounds the AWAIT with SUMMARY_TIMEOUT_SECONDS:

  * the background summary task swallows the timeout (the file's embeddings are already
    stored and usable; the summary is best-effort);
  * the /summarize endpoint returns a retryable 503, never a hang and never a fake/empty
    summary.

Honest limit (see app/config.py): wait_for frees the request, not the worker thread. This
proves the request is bounded; the thread-level bound is an LLM-client control left to the
operator. The tests use a fake summarizer that SLEEPS, so 'the worker overran' is arranged.
"""

import asyncio
import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes import document_routes as dr
from tests.utils.test_replace_not_accumulate import FakeStore

_SECRET = "testsecret"


def _slow_summarize(seconds):
    def _fn(llm, grouped):
        time.sleep(seconds)
        return [{"file_id": fid, "summary": "s", "chunk_count": 1} for fid in grouped]
    return _fn


def _fast_summarize(llm, grouped):
    return [{"file_id": fid, "summary": "s", "chunk_count": 1} for fid in grouped]


# --- the helper: the actual bound -------------------------------------------------------


def test_slow_worker_is_bounded_and_a_fast_one_is_not(monkeypatch):
    monkeypatch.setattr(dr, "summarize_files", _slow_summarize(1.0))
    monkeypatch.setattr(dr, "SUMMARY_TIMEOUT_SECONDS", 0.2, raising=False)

    async def run(fn_grouped):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as pool:
            return await dr._summarize_within_timeout(loop, pool, object(), fn_grouped)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run({"f1": [Document(page_content="x")]}))

    # A worker that finishes within the bound returns normally (no over-rejection).
    monkeypatch.setattr(dr, "summarize_files", _fast_summarize)
    out = asyncio.run(run({"f1": [Document(page_content="x")]}))
    assert out == [{"file_id": "f1", "summary": "s", "chunk_count": 1}]


# --- background task: timeout is swallowed, embeddings preserved ------------------------


def test_background_summary_timeout_does_not_raise(monkeypatch):
    monkeypatch.setattr(dr, "SUM_UP_KNOWLEDGE_FILES", True, raising=False)
    monkeypatch.setattr(dr, "SUMMARY_TIMEOUT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(dr, "summarize_files", _slow_summarize(1.0))
    persisted = []
    async def _upsert(*a, **k):
        persisted.append(a)
    monkeypatch.setattr(dr, "upsert_file_summary", _upsert)

    async def run():
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Must NOT raise: a background task has no caller, and the embeddings are
            # already stored, so a summary timeout is logged and dropped.
            await dr._generate_summary_background("f1", "u", [Document(page_content="x")], pool, object())

    asyncio.run(run())
    assert persisted == [], "a timed-out summary must not be persisted"


# --- /summarize endpoint: retryable 503, not a hang -------------------------------------


class _FakeAsyncStore(FakeStore):
    """FakeStore (real write methods) + the grouping call the /summarize route makes.
    Subclasses AsyncPgVector transitively, so the route's async branch is taken and the
    combined-summary embed at the end of the route uses FakeStore.aadd_documents."""
    async def get_documents_grouped_by_file_id(self, user_id=None, executor=None):
        return {"f1": [Document(page_content="body", metadata={"file_id": "f1"})]}
    async def delete(self, ids=None, collection_only=False, user_id=None,
                     document_origin_type=None, subscription_id=None, text_source=None,
                     executor=None, **_):
        return await super().delete(ids=ids)


@pytest.fixture()
def client(monkeypatch):
    os.environ["JWT_SECRET"] = _SECRET
    monkeypatch.setattr(dr, "vector_store", _FakeAsyncStore())
    monkeypatch.setattr(dr, "llm", object())  # not None -> route proceeds
    async def _no_cache(user_id):
        return []
    monkeypatch.setattr(dr, "get_summaries_by_user", _no_cache)
    async def _upsert(*a, **k):
        return None
    monkeypatch.setattr(dr, "upsert_file_summary", _upsert)
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


def _hdr(ent=("kbA",), act=("read", "write")):
    os.environ["JWT_SECRET"] = _SECRET
    return {"Authorization": "Bearer " + jwt.encode(
        {"id": "u", "tid": "tA", "ent": list(ent), "act": list(act),
         "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
        _SECRET, algorithm="HS256")}


def test_summarize_endpoint_returns_503_on_timeout_not_a_hang(client, monkeypatch):
    monkeypatch.setattr(dr, "SUMMARY_TIMEOUT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(dr, "summarize_files", _slow_summarize(1.0))
    r = client.post("/summarize/kbA", data={"knowledge_id": "kbA", "file_id": "f1"}, headers=_hdr())
    assert r.status_code == 503, r.text
    assert "timed out" in r.json()["detail"].lower()


def test_summarize_endpoint_succeeds_within_the_bound(client, monkeypatch):
    monkeypatch.setattr(dr, "SUMMARY_TIMEOUT_SECONDS", 5.0, raising=False)
    monkeypatch.setattr(dr, "summarize_files", _fast_summarize)
    r = client.post("/summarize/kbA", data={"knowledge_id": "kbA", "file_id": "f1"}, headers=_hdr())
    assert r.status_code == 200, r.text

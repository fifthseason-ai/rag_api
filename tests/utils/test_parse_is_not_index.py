"""PARSE != INDEX: the intake receipt reports the two as separate states (KC-FILES-1).

WHY. Every intake route returned `extraction` -- what the loader READ -- and HTTP 200. Nothing
said what the table then HELD, so a consumer could only read "200" as "indexed". A write that
landed short (the store accepted the call but holds fewer of this write's chunks than were
prepared) was indistinguishable from a complete one.

The `index` block is read back from the store, keyed on this write's `ingest_id`:
  indexed     chunks_confirmed == chunks_prepared > 0
  partial     any other confirmed count -- never `indexed`
  unverified  the store could not be read back: UNKNOWN, never promoted to `indexed`

SYNTHETIC documents only (built here: txt, xlsx, zip, encrypted pdf); no real source.

The FakeStore tests run everywhere. The @needs_pg tests drive the real route against a real
pgvector table at the production batch size, so the read-back SQL itself is proven, not
modelled.
"""

import datetime
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes
from tests.utils.test_replace_not_accumulate import FakeRow, FakeStore

_SECRET = "testsecret"
FID = "kc-files-1"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
TEXT = "\n\n".join(f"Paragraph {i}. " + ("synthetic words " * 40) for i in range(6))

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the real read-back",
)


def _hdr():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"], "act": ["read", "write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _xlsx():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Revenue"
    ws.append(["region", "quarter", "amount"])
    for i in range(12):
        ws.append([f"region-{i}", f"Q{i % 4 + 1}", 1000 + i])
    wb.create_sheet("Notes").append(["synthetic workbook for KC-FILES-1"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("inner.txt", "a text file inside an archive")
    return buf.getvalue()


def _encrypted_pdf():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.encrypt(user_password="secret", owner_password="owner")
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class ShortStore(FakeStore):
    """Accepts the whole write, keeps all but the LAST chunk, and returns ids for all of
    them -- a store that lands short without raising. The fixture can express the failure:
    the rows it holds really are fewer than were prepared."""

    async def aadd_documents(self, docs, ids=None, executor=None):
        self.calls.append("insert")
        for d in docs[:-1]:
            self.rows.append(FakeRow(ids[0] if ids else None, d.page_content, d.metadata))
        return ids


class BlindStore(FakeStore):
    """Stores faithfully but cannot be read back."""

    async def count_rows_for_ingest(self, file_id, ingest_id, executor=None):
        raise ConnectionError("read-back unavailable")


def _client(monkeypatch, store, batch=0):
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", batch, raising=False)
    return TestClient(app)


def _post(client, route, name, content, mime):
    if route == "/local/embed":
        from app.config import RAG_UPLOAD_DIR

        os.makedirs(os.path.join(RAG_UPLOAD_DIR, "userA"), exist_ok=True)
        with open(os.path.join(RAG_UPLOAD_DIR, "userA", name), "wb") as f:
            f.write(content)
        return client.post(
            "/local/embed",
            json={"filepath": f"userA/{name}", "filename": name,
                  "file_content_type": mime, "file_id": FID},
            params={"entity_id": "userA"}, headers=_hdr(),
        )
    field = "file" if route == "/embed" else "uploaded_file"
    return client.post(route, data={"file_id": FID, "entity_id": "userA"}, headers=_hdr(),
                       files={field: (name, io.BytesIO(content), mime)})


def _stored(store):
    return [r for r in store.rows if r.custom_id == FID]


ROUTES = ["/embed", "/embed-upload", "/local/embed"]
DOCS = [("doc.txt", TEXT.encode(), "text/plain"), ("book.xlsx", None, XLSX_MIME)]


# --- the positive case: indexed means the rows are really there ------------------------

@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name,content,mime", DOCS, ids=["txt", "xlsx"])
def test_a_complete_write_reads_indexed_with_the_real_row_count(monkeypatch, route, name, content, mime):
    store = FakeStore()
    r = _post(_client(monkeypatch, store), route, name, content or _xlsx(), mime)
    assert r.status_code == 200, r.text
    body = r.json()
    rows = _stored(store)
    assert rows, "precondition: the write stored rows"
    idx = body["index"]
    assert idx["status"] == "indexed", idx
    assert idx["chunks_confirmed"] == idx["chunks_prepared"] == len(rows)
    # parse and index are SEPARATE blocks; one is not derived from the other
    assert body["extraction"]["status"] in ("complete", "partial")
    assert {r.metadata["ingest_id"] for r in rows} == {idx["ingest_id"]}


# --- the red-first case: a short store must never read as indexed ----------------------

@pytest.mark.parametrize("route", ROUTES)
def test_a_short_write_is_partial_never_indexed(monkeypatch, route):
    store = ShortStore()
    r = _post(_client(monkeypatch, store), route, "doc.txt", TEXT.encode(), "text/plain")
    assert r.status_code == 200, r.text  # additive: the status code is unchanged
    idx = r.json()["index"]
    held = len(_stored(store))
    assert idx["chunks_prepared"] > held >= 1, (idx, held)  # the fixture really is short
    assert idx["status"] == "partial", idx
    assert idx["chunks_confirmed"] == held


def test_an_unreadable_store_is_unverified_never_indexed(monkeypatch):
    store = BlindStore()
    r = _post(_client(monkeypatch, store), "/embed", "doc.txt", TEXT.encode(), "text/plain")
    assert r.status_code == 200, r.text
    idx = r.json()["index"]
    assert _stored(store), "precondition: the rows WERE stored -- only the read-back failed"
    assert idx["status"] == "unverified" and idx["chunks_confirmed"] is None, idx


def test_the_count_is_scoped_to_this_write_not_the_file(monkeypatch):
    """Additive default keeps the earlier version: its rows must not be counted as ours."""
    store = FakeStore()
    client = _client(monkeypatch, store)
    assert _post(client, "/embed", "doc.txt", TEXT.encode(), "text/plain").status_code == 200
    r = _post(client, "/embed", "short.txt", b"one short synthetic line", "text/plain")
    idx = r.json()["index"]
    assert len(_stored(store)) > idx["chunks_prepared"], "precondition: two versions held"
    assert idx["status"] == "indexed" and idx["chunks_confirmed"] == idx["chunks_prepared"] == 1


# --- negatives: nothing indexed, and nothing claims to be ------------------------------

@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name,content,mime", [
    ("empty.txt", b"   \n\n  ", "text/plain"),
    ("archive.zip", None, "application/zip"),
    ("locked.pdf", None, "application/pdf"),
])
def test_empty_unsupported_and_encrypted_are_refused_and_never_indexed(monkeypatch, route, name, content, mime):
    if content is None:
        content = _zip() if name.endswith(".zip") else _encrypted_pdf()
    store = FakeStore()
    r = _post(_client(monkeypatch, store), route, name, content, mime)
    assert r.status_code >= 400, (r.status_code, r.text)
    assert "indexed" not in r.text, r.text
    assert "index" not in r.json()
    assert _stored(store) == []


# --- real pgvector: the read-back SQL itself --------------------------------------------

def _real_store(monkeypatch, collection):
    import psycopg2
    from app.services.vector_store.factory import get_vector_store
    from tests.utils.test_empty_entitlement_query_path import _DetEmb

    raw = PG_DSN.replace("postgresql+psycopg2://", "postgresql://")
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()
    dsn = PG_DSN.replace("postgresql://", "postgresql+psycopg2://", 1)
    store = get_vector_store(dsn, _DetEmb(), collection, mode="async")
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store
    if store.create_extension:
        store.create_vector_extension()
    store.EmbeddingStore, store.CollectionStore = _get_embedding_collection_store(
        store._embedding_length, use_jsonb=store.use_jsonb
    )
    store.create_tables_if_not_exists()
    store.create_collection()
    return store


def _real_rows(store):
    from app.services.vector_store.extended_pg_vector import ExtendedPgVector

    return len(ExtendedPgVector.get_row_uuids(store, FID))  # sync read, straight from the table


@needs_pg
@pytest.mark.parametrize("batch", [0, 500])
def test_real_pgvector_indexed_count_matches_the_table(monkeypatch, batch):
    store = _real_store(monkeypatch, "kc_files_1")
    client = _client(monkeypatch, store, batch=batch)
    r = _post(client, "/embed", "doc.txt", TEXT.encode(), "text/plain")
    assert r.status_code == 200, r.text
    idx = r.json()["index"]
    assert idx["status"] == "indexed", idx
    assert idx["chunks_confirmed"] == idx["chunks_prepared"] == _real_rows(store) > 1
    # a second additive write is counted on its own
    r2 = _post(client, "/embed", "short.txt", b"one short synthetic line", "text/plain")
    idx2 = r2.json()["index"]
    assert idx2["status"] == "indexed" and idx2["chunks_confirmed"] == 1
    assert _real_rows(store) == idx["chunks_confirmed"] + 1


@needs_pg
def test_real_pgvector_short_write_reads_partial(monkeypatch):
    store = _real_store(monkeypatch, "kc_files_1_short")
    real_add = type(store).aadd_documents

    async def short_add(self, docs, ids=None, executor=None, **kw):
        return await real_add(self, docs[:-1], ids=(ids or [])[:-1] or None,
                              executor=executor, **kw) and ids

    monkeypatch.setattr(type(store), "aadd_documents", short_add)
    r = _post(_client(monkeypatch, store), "/embed", "doc.txt", TEXT.encode(), "text/plain")
    assert r.status_code == 200, r.text
    idx = r.json()["index"]
    held = _real_rows(store)
    assert idx["chunks_prepared"] == held + 1 and held >= 1, (idx, held)
    assert idx["status"] == "partial" and idx["chunks_confirmed"] == held, idx

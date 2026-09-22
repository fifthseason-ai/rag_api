"""P06-2 INGESTION & RECEIPTS: per-source receipt, retry idempotency, duplicate
canonicalisation and negative-case zero-rows -- proven through the REAL intake path
(`POST /embed`) against a REAL pgvector table, with SQL read-back as the ground truth.

WHY THIS FILE EXISTS. The KC-FILES-1 receipt (`index` block) and the F04 rollback /
F03 replacement primitives are each proven elsewhere, but the KNOWLEDGE/ASTRA ingestion
card (P06-2) asks four things those files do not assert TOGETHER at the table:

  1. PER-SOURCE RECEIPT + IDENTITY/VERSION/PERMISSION SURVIVAL. Every stored row must
     carry the source identity (`file_id`, `filename`), the version (`ingest_id`) and the
     permission axes (`user_id` = resolved entity, `tenant_id`). Proven by SELECTing the
     rows back, NOT by trusting the response body -- "the body said so" and "the table
     holds it" are different claims.

  2. RETRY. A transient DB fault must resume WITHOUT duplicating rows. The fault here is
     a REAL psycopg2 error: a BEFORE INSERT trigger on `langchain_pg_embedding` that
     `RAISE`s once a partial batch has already committed, so the route's rollback has real
     committed rows to remove. Idempotency is shown by a ROW COUNT after the retry, not
     asserted. NON-VACUITY: with the rollback neutered, the same fault leaves residue and
     the retry DOUBLES -- the control reddens.

  3. DUPLICATE. The same source/version ingested twice must yield ONE canonical record,
     never two. rag_api's canonicalisation is `replace=true` (capture-before-insert,
     delete-after): re-ingesting supersedes, leaving one `ingest_id`. Proven by a COUNT
     and a DISTINCT-ingest_id count. NON-VACUITY: with the supersede-delete neutered the
     count goes to TWO versions -- the control reddens. (The additive NO-replace default
     is deliberate and pinned in test_replace_not_accumulate.py; it is NOT re-litigated
     here -- this file proves the canonical path a Knowledge ingester must use.)

  4. SHAPE COVERAGE + ZERO ROWS ON NEGATIVES. multi-sheet xlsx -> locator_kind "sheet"
     and rows carrying `page_name`; zip/png -> 422 unsupported; encrypted -> 422 encrypted;
     and on EVERY negative the table holds ZERO rows -- verified by SQL, because "422
     returned" and "nothing was written" are different claims.

The `index` block (indexed|partial|unverified, ingest_id, chunks_prepared/confirmed) is
asserted on the success cases and MUST BE ABSENT on the refused ones.

EVERYTHING HERE IS SYNTHETIC. The sources are built in-process (txt, multi-sheet xlsx,
zip, png, encrypted PDF); there is no real Knowledge original on this host (blocked on
A3 Box / A4 Graph credentials). SYNTHETIC proof of the ingestion PATH is NOT proof of
real-source ingestion.

Needs a Postgres+pgvector. RAG_TEST_PG_DSN selects it; RAG_TEST_PG_REQUIRED=1 turns "no
DSN" from a skip into an error so these provably ran.
"""

import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import psycopg2
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector

# Reuse the canonical real-pgvector store builder and the synthetic source builders that
# other KC-FILES-1 / unsupported-format suites already pin, so this file proves NEW
# properties rather than re-deriving fixtures.
from tests.utils.test_parse_is_not_index import (
    PG_DSN,
    needs_pg,
    _real_store,
    _xlsx,
    _zip,
    _encrypted_pdf,
)
from tests.utils.test_unsupported_formats import make_png

_SECRET = "testsecret"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Distinct identity/version/permission values so a leak or a mis-copy is visible in SQL.
ENTITY = "ent-knowledge-1"          # resolved entity -> stored cmetadata.user_id
TENANT = "tenant-vivaldi"           # token tid       -> stored cmetadata.tenant_id
FILENAME = "astra-brief.txt"

# Long enough to split into several chunks at CHUNK_SIZE=1500, so a partial-batch commit
# is possible and "rows removed on rollback" is a real event, not a single-row no-op.
_SECTION = (
    "Section {n}. This is synthetic Knowledge-lane ingestion content authored in-process "
    "for the P06-2 receipt proof. It carries no real source material. "
)
TEXT = "\n\n".join((_SECTION.format(n=i) + ("synthetic words " * 90)) for i in range(6))


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _hdr(entity=ENTITY, tid=TENANT, act=("read", "write")):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "uploader-user",
        "tid": tid,
        "ent": [entity],
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _client(monkeypatch, store, batch=0):
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", batch, raising=False)
    return TestClient(app)


def _embed(client, file_id, name, content, mime, replace=None, entity=ENTITY, tid=TENANT):
    data = {"file_id": file_id, "entity_id": entity}
    if replace is not None:
        data["replace"] = str(replace).lower()
    return client.post(
        "/embed", data=data, headers=_hdr(entity=entity, tid=tid),
        files={"file": (name, io.BytesIO(content), mime)},
    )


# --- SQL ground truth: read the table, never the response body -------------------------

def _rows_meta(file_id):
    """Every stored row's identity/version/permission fields, straight from the table."""
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute(
            "SELECT cmetadata->>'file_id', cmetadata->>'user_id', cmetadata->>'tenant_id', "
            "cmetadata->>'ingest_id', cmetadata->>'filename', cmetadata->>'digest', "
            "cmetadata->>'page_name' "
            "FROM langchain_pg_embedding WHERE custom_id = %s",
            (file_id,),
        )
        cols = ("file_id", "user_id", "tenant_id", "ingest_id", "filename", "digest", "page_name")
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _count(file_id):
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM langchain_pg_embedding WHERE custom_id = %s", (file_id,)
        )
        return cur.fetchone()[0]


# --- a REAL transient fault: a plpgsql trigger that RAISEs, armed out-of-band ----------

def _install_fault(fire_at):
    """Arm a BEFORE INSERT trigger that RAISEs on the `fire_at`-th row inserted into
    langchain_pg_embedding. `seen` is committed by every batch that succeeds, so once a
    partial batch has committed its rows, the next batch's insert trips the RAISE -- the
    route then has genuinely-committed rows to roll back. A real psycopg2 error (SQLSTATE
    P0001 -> psycopg2.errors.RaiseException), not a spoofed exception class."""
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("DROP TRIGGER IF EXISTS _files01_fault_trg ON langchain_pg_embedding")
        cur.execute("DROP TABLE IF EXISTS _files01_fault")
        cur.execute("CREATE TABLE _files01_fault(id int PRIMARY KEY, armed boolean NOT NULL, "
                    "fire_at int NOT NULL, seen int NOT NULL DEFAULT 0)")
        cur.execute("INSERT INTO _files01_fault(id, armed, fire_at) VALUES (1, true, %s)", (fire_at,))
        cur.execute(
            """
            CREATE OR REPLACE FUNCTION _files01_fault_fn() RETURNS trigger AS $$
            DECLARE a boolean; f int; s int;
            BEGIN
              SELECT armed, fire_at, seen INTO a, f, s FROM _files01_fault WHERE id = 1 FOR UPDATE;
              IF a THEN
                s := s + 1;
                UPDATE _files01_fault SET seen = s WHERE id = 1;
                IF s >= f THEN
                  RAISE EXCEPTION 'FILES01-TRANSIENT-FAULT injected at row % (seen=%)', NEW.custom_id, s;
                END IF;
              END IF;
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        cur.execute("CREATE TRIGGER _files01_fault_trg BEFORE INSERT ON langchain_pg_embedding "
                    "FOR EACH ROW EXECUTE FUNCTION _files01_fault_fn()")
        c.commit()


def _disarm_fault():
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("UPDATE _files01_fault SET armed = false WHERE id = 1")
        c.commit()


def _committed_seen():
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT seen FROM _files01_fault WHERE id = 1")
        return cur.fetchone()[0]


def _remove_fault():
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("DROP TRIGGER IF EXISTS _files01_fault_trg ON langchain_pg_embedding")
        cur.execute("DROP FUNCTION IF EXISTS _files01_fault_fn()")
        cur.execute("DROP TABLE IF EXISTS _files01_fault")
        c.commit()


# =======================================================================================
# GOAL 1 -- per-source receipt: identity + version + permissions survive INTO the table
# =======================================================================================

@needs_pg
def test_source_identity_version_and_permissions_survive_into_stored_rows(monkeypatch):
    store = _real_store(monkeypatch, "p062_receipt")
    client = _client(monkeypatch, store, batch=500)  # production batch path
    fid = "kn-src-brief"

    r = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain")
    assert r.status_code == 200, r.text
    body = r.json()

    # the index block IS the "indexed" leg of the receipt, read back from the store
    idx = body["index"]
    assert idx["status"] == "indexed", idx
    assert idx["chunks_confirmed"] == idx["chunks_prepared"] > 1, idx

    # GROUND TRUTH: read the rows back and prove the fields are on EVERY row, not the body
    rows = _rows_meta(fid)
    assert rows, "precondition: rows were stored"
    assert _count(fid) == idx["chunks_confirmed"], "table count must equal the receipt"
    for row in rows:
        assert row["file_id"] == fid                    # source identity
        assert row["filename"] == FILENAME              # source identity (display)
        assert row["user_id"] == ENTITY                 # permission: owning entity
        assert row["tenant_id"] == TENANT               # permission: tenant tag
        assert row["ingest_id"] == idx["ingest_id"]     # version: this write's id
        assert row["digest"], "every row carries a content digest"
    # version axis is single-valued for one write
    assert {row["ingest_id"] for row in rows} == {idx["ingest_id"]}


# =======================================================================================
# GOAL 2 -- RETRY: a real transient fault resumes without duplicating rows
# =======================================================================================

@needs_pg
def test_retry_after_a_real_transient_fault_does_not_duplicate_rows(monkeypatch):
    # ONE store / ONE global binding: _real_store DROPs the shared table and rebinds
    # document_routes.vector_store, so a second _real_store call in the same test would
    # wipe this one and point the route (and its rollback) at the wrong object.
    store = _real_store(monkeypatch, "p062_retry")
    client = _client(monkeypatch, store, batch=2)  # small batches: batch 1 commits, 2 fails
    fid = "kn-retry-src"

    # baseline BEFORE the fault is installed: how many chunks a CLEAN ingest produces.
    # Its rows have a different custom_id, so the fault's row counter never sees them and
    # the fid rollback (delete-by-file_id) never touches them.
    assert _embed(client, "kn-baseline", FILENAME, TEXT.encode(), "text/plain").status_code == 200
    clean_count = _count("kn-baseline")
    assert clean_count > 2, "fixture must split into several batches for this proof"

    _install_fault(fire_at=3)  # fires on the 3rd row inserted AFTER install
    try:
        r1 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain")
        # a DB RAISE is a service fault -> retryable 503, attributed, never a 200 fake-success
        assert r1.status_code == 503, (r1.status_code, r1.text)
        assert "not with your file" in r1.text, r1.text
        assert "FILES01-TRANSIENT-FAULT" not in r1.text, "the raw DB error leaked to the caller"
        # partial batch DID commit, then the route rolled it back: table is clean
        assert _committed_seen() >= 2, "precondition: a partial batch really committed"
        assert _count(fid) == 0, "rollback must remove the partially-committed rows"

        _disarm_fault()
        r2 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain")
        assert r2.status_code == 200, r2.text
        idx = r2.json()["index"]
        assert idx["status"] == "indexed", idx
        # THE IDEMPOTENCY CLAIM, by count: the retry holds exactly one clean version
        assert _count(fid) == clean_count == idx["chunks_confirmed"], (
            _count(fid), clean_count, idx,
        )
    finally:
        _remove_fault()


@needs_pg
def test_retry_control_without_rollback_leaves_residue_and_doubles(monkeypatch):
    """NON-VACUITY for the retry guard: neuter the route's rollback delete and the SAME
    fault leaves the partial batch behind, so the retry accumulates. If this did not
    redden, the test above would prove nothing."""
    store = _real_store(monkeypatch, "p062_retry_ctrl")
    client = _client(monkeypatch, store, batch=2)
    fid = "kn-retry-ctrl"

    assert _embed(client, "kn-ctrl-base", FILENAME, TEXT.encode(), "text/plain").status_code == 200
    clean_count = _count("kn-ctrl-base")

    async def _noop_delete(*a, **k):  # the non-replace rollback seam
        return None

    monkeypatch.setattr(store, "delete", _noop_delete)

    _install_fault(fire_at=3)
    try:
        r1 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain")
        assert r1.status_code == 503, r1.text
        residue = _count(fid)
        assert residue >= 2, ("control: with rollback neutered the partial batch survives", residue)

        _disarm_fault()
        r2 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain")
        assert r2.status_code == 200, r2.text
        # the residue is NOT cleaned, so the retry duplicates: strictly more than one version
        assert _count(fid) == residue + clean_count > clean_count, (_count(fid), residue, clean_count)
    finally:
        _remove_fault()


# =======================================================================================
# GOAL 3 -- DUPLICATE: same source/version twice = ONE canonical record
# =======================================================================================

@needs_pg
def test_the_same_source_version_ingested_twice_is_one_canonical_record(monkeypatch):
    store = _real_store(monkeypatch, "p062_dup")
    client = _client(monkeypatch, store, batch=500)
    fid = "kn-dup-src"

    r1 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain", replace=True)
    assert r1.status_code == 200, r1.text
    first = _count(fid)
    first_ingest = r1.json()["index"]["ingest_id"]
    assert first > 1

    # ingest the SAME source/version again, the canonical way (replace)
    r2 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain", replace=True)
    assert r2.status_code == 200, r2.text
    second_ingest = r2.json()["index"]["ingest_id"]

    rows = _rows_meta(fid)
    # ONE canonical record: the count did not grow, and only the NEW version's rows remain
    assert _count(fid) == first, ("duplicate must not accumulate", _count(fid), first)
    assert {row["ingest_id"] for row in rows} == {second_ingest}, "only the latest version survives"
    assert first_ingest != second_ingest, "each ingest is its own version id"
    # the replacement receipt says it superseded the earlier version completely
    assert r2.json()["replacement"]["status"] == "complete", r2.json()["replacement"]


@needs_pg
def test_duplicate_control_without_supersede_delete_yields_two(monkeypatch):
    """NON-VACUITY for the duplicate guard: neuter the supersede-delete and the same
    double-ingest leaves BOTH versions -- exactly the two-record outcome the guard
    prevents."""
    store = _real_store(monkeypatch, "p062_dup_ctrl")
    client = _client(monkeypatch, store, batch=500)
    fid = "kn-dup-ctrl"

    r1 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain", replace=True)
    assert r1.status_code == 200, r1.text
    first = _count(fid)

    async def _noop_rows(*a, **k):  # the supersede-delete seam
        return 0

    monkeypatch.setattr(store, "delete_rows_by_uuid", _noop_rows)

    r2 = _embed(client, fid, FILENAME, TEXT.encode(), "text/plain", replace=True)
    assert r2.status_code == 200, r2.text
    # both versions now present -> TWO records, and two distinct ingest ids
    assert _count(fid) == 2 * first, ("control: without the delete, both versions persist", _count(fid), first)
    assert len({row["ingest_id"] for row in _rows_meta(fid)}) == 2
    assert r2.json()["replacement"]["status"] == "incomplete", r2.json()["replacement"]


# =======================================================================================
# GOAL 4 -- shape coverage + ZERO rows on every negative, by SQL
# =======================================================================================

@needs_pg
def test_multisheet_xlsx_indexes_with_a_sheet_locator(monkeypatch):
    store = _real_store(monkeypatch, "p062_xlsx")
    client = _client(monkeypatch, store, batch=500)
    fid = "kn-xlsx-src"

    r = _embed(client, fid, "book.xlsx", _xlsx(), XLSX_MIME)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extraction"]["locator_kind"] == "sheet", body["extraction"]
    assert body["index"]["status"] == "indexed", body["index"]

    rows = _rows_meta(fid)
    assert rows and _count(fid) == body["index"]["chunks_confirmed"]
    # the sheet locator survives onto the stored rows (page_name), and identity too
    assert any(row["page_name"] for row in rows), "no sheet locator on any stored row"
    assert {row["user_id"] for row in rows} == {ENTITY}
    assert {row["tenant_id"] for row in rows} == {TENANT}


@needs_pg
@pytest.mark.parametrize(
    "name,mime,verdict",
    [
        ("archive.zip", "application/zip", "unsupported"),
        ("photo.png", "image/png", "unsupported"),
        ("locked.pdf", "application/pdf", "encrypted"),
    ],
    ids=["zip", "png", "encrypted"],
)
def test_negatives_are_refused_422_and_write_zero_rows(monkeypatch, tmp_path, name, mime, verdict):
    store = _real_store(monkeypatch, "p062_neg")
    client = _client(monkeypatch, store, batch=500)
    fid = f"kn-neg-{verdict}"

    if name.endswith(".zip"):
        content = _zip()
    elif name.endswith(".png"):
        png_path = tmp_path / "photo.png"
        make_png(str(png_path))
        content = png_path.read_bytes()
    else:  # a genuinely password-protected file -> the encrypted verdict
        content = _encrypted_pdf()

    r = _embed(client, fid, name, content, mime)
    assert r.status_code == 422, (r.status_code, r.text)
    detail = r.json()["detail"]
    assert detail["extraction"]["status"] == "unsupported", detail
    assert detail["extraction"]["verdict"] == verdict, detail
    # NO index block on a refusal -- absent, never a fabricated "indexed"
    assert "index" not in r.json(), r.json()
    # and, the load-bearing claim, verified at the table not from the 422:
    assert _count(fid) == 0, "a refused source must write ZERO rows"

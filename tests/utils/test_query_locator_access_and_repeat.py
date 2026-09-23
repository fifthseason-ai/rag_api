"""Antonio's retrieval path at the PRODUCER (PACKET-1, C2P1): the new per-unit locators reach
the wire through the exact /query routes Core calls, are never fabricated, never cross a file or
entity boundary, and the SAME question asked twice returns a BYTE-IDENTICAL ordered result.

Richard's goal, verbatim: "design self checks and verify that antonio can access it and if asked
the same question 2 times, still surfaces the same answer". This file is the rag_api half of it.
Core's consumer (readPassageLocator) and the live Antonio journey are NOT proven here.

Core's request shape, read from release/richard-vibe 091d9729
(api/server/services/Knowledge/retrieval.js, api/app/clients/tools/util/fileSearch.js):
  knowledge leg  POST /query/{entityId}  {query, k: legK, args: {cohere: false}}
  file leg       POST /query             {file_id, query, k: legK}      (one call PER file)
  headers        Authorization: Bearer <KSPT-01 token: tid, ent=[...], act=['read']>
  legK           max(DEFAULT_QUERY_K 15 * 3, MIN_RETRIEVAL_K 40) = 45

Two retrieval paths are exercised:
  default  the app's configured defaults (HYBRID_SEARCH_ENABLED and RERANK_ENABLED both default
           True in app/config.py) with a LIVE keyword leg (document_tsv added the way TEMPO
           migration 10081 shapes it, as test_both_legs_expected_passage_and_version does) and the
           Bedrock rerank CALL replaced by a raise, so the real rerank() fallback runs -- what
           production does when the Bedrock rerank errors. A live Cohere reorder is an external
           boundary this suite cannot reach.
  dense    HYBRID_SEARCH_ENABLED=False, RERANK_ENABLED=False: the vector store alone.

NOT COVERED, on purpose (see the C2P1 FILES finding): EXACT-TIE ordering. The store orders by
`distance` only, with no tie-break, so rows at the SAME distance come back in heap/plan order --
stable on a serial plan, NOT stable on a parallel one. The repeat tests below use distinct texts
and assert exact equality; they do not claim tie stability.

SYNTHETIC fixtures only, built at test time. Needs a Postgres with pgvector: RAG_TEST_PG_DSN
selects it and RAG_TEST_PG_REQUIRED=1 turns a missing DSN into an ERROR (CI sets both); with
neither set every test here SKIPS. Reuses the existing real-store harness (needs_pg, _real_store,
_client from test_parse_is_not_index; _DetEmb, _sqlalchemy_dsn from
test_empty_entitlement_query_path) -- no second harness.
"""

import datetime
import io
import os
import zipfile

import jwt
import psycopg2
import pytest
from fastapi.testclient import TestClient

from tests.utils.test_docx_reading_order import DOCX_MIME, W_NS
from tests.utils.test_empty_entitlement_query_path import _DetEmb, _sqlalchemy_dsn
from tests.utils.test_parse_is_not_index import PG_DSN, _client, _real_store, needs_pg

_SECRET = "testsecret"
TENANT = "tenantA"
USER_A = "userA"
USER_B = "userB"
CORE_K = 45
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MD_MIME = "text/markdown"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

DOCX_TOKEN = "C2P1QDOCXTOKEN"
HDR_TOKEN = "C2P1QHEADERTOKEN"
FTR_TOKEN = "C2P1QFOOTERTOKEN"
XLSX_TOKEN = "C2P1QXLSXTOKEN"
MD_TOKEN = "C2P1QMDTOKEN"
PRE_TOKEN = "C2P1QPREAMBLETOKEN"

F_DOCX, F_XLSX, F_MD = "c2p1-q-docx", "c2p1-q-xlsx", "c2p1-q-md"


# --------------------------------------------------------------------------- SYNTHETIC builders
def docx_with_header_footer_SYNTHETIC():
    """Three body paragraphs (block_index 0,1,2; the answer is 1) plus a header and a footer
    part, whose units must carry NO block_index."""
    ct = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    body = (
        "<w:p><w:r><w:t>Introduction paragraph about general company matters.</w:t></w:r></w:p>"
        f"<w:p><w:r><w:t>The {DOCX_TOKEN} quarterly revenue figure is 4200000 euros.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Closing remarks, disclaimers and appendix pointers.</w:t></w:r></w:p>"
    )
    document = (
        f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>{body}'
        '<w:sectPr><w:headerReference w:type="default" r:id="rIdH"/>'
        '<w:footerReference w:type="default" r:id="rIdF"/></w:sectPr>'
        "</w:body></w:document>"
    )
    header = f'<?xml version="1.0"?><w:hdr xmlns:w="{W_NS}"><w:p><w:r><w:t>{HDR_TOKEN} confidential banner</w:t></w:r></w:p></w:hdr>'
    footer = f'<?xml version="1.0"?><w:ftr xmlns:w="{W_NS}"><w:p><w:r><w:t>{FTR_TOKEN} page footer</w:t></w:r></w:p></w:ftr>'
    drels = (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdH" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>'
        '<Relationship Id="rIdF" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
        "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", drels)
        z.writestr("word/header1.xml", header)
        z.writestr("word/footer1.xml", footer)
    return buf.getvalue()


def xlsx_with_blank_sheet_SYNTHETIC():
    """Revenue: header row 1 + two data rows (occupied extent A1:C3). Blank: no cell at all."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Revenue"
    for row in (["Region", "Metric", "Value"], ["EMEA", XLSX_TOKEN, 4200000], ["AMER", "revenue", 3100000]):
        ws.append(row)
    wb.create_sheet("Blank")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def md_with_preamble_SYNTHETIC():
    return (
        f"{PRE_TOKEN} text before any heading.\n\n"
        "# Overview\n\nGeneral introduction to the report.\n\n"
        f"## Revenue\n\nThe {MD_TOKEN} quarterly revenue figure is 4200000 euros.\n\n"
        "## Costs\n\nMiscellaneous cost commentary.\n"
    ).encode("utf-8")


FIXTURES = {
    "docx": (F_DOCX, "report.docx", DOCX_MIME, docx_with_header_footer_SYNTHETIC),
    "xlsx": (F_XLSX, "book.xlsx", XLSX_MIME, xlsx_with_blank_sheet_SYNTHETIC),
    "md": (F_MD, "notes.md", MD_MIME, md_with_preamble_SYNTHETIC),
}


# --------------------------------------------------------------------------- harness glue
def _token(ent=(USER_A,), act=("read",), secret=_SECRET, tid=TENANT, hours=1):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": ent[0] if ent else "nobody", "tid": tid, "ent": list(ent), "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _core_hdr(ent=(USER_A,)):
    """Core's header: Bearer KSPT-01 token (read-only act) + JSON content type."""
    return {"Authorization": f"Bearer {_token(ent)}", "Content-Type": "application/json"}


def _write_hdr():
    return {"Authorization": f"Bearer {_token((USER_A,), act=('read', 'write'))}"}


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _add_keyword_column():
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_tsv ON langchain_pg_embedding USING gin (document_tsv)"
        )
        c.commit()


def _select_path(monkeypatch, path):
    """default: the app's defaults + a live keyword leg + Bedrock rerank failing (real fallback).
    dense: hybrid and rerank off."""
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services import reranker
    from app.services.database import PSQLDatabase

    if path == "dense":
        monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", False)
        monkeypatch.setattr(dr, "RERANK_ENABLED", False)
        return
    assert dr.HYBRID_SEARCH_ENABLED is True and dr.RERANK_ENABLED is True, (
        "precondition: the app defaults are hybrid ON and rerank ON",
        dr.HYBRID_SEARCH_ENABLED, dr.RERANK_ENABLED)
    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", _raw_dsn(), raising=True)
    real_kw = dr.keyword_search

    async def _kw(*a, **k):
        # TestClient runs each request on its own loop: rebuild the asyncpg pool per call
        # (same pattern as test_both_legs_expected_passage_and_version / F-ENTITLEMENT-FUSED).
        try:
            return await real_kw(*a, **k)
        finally:
            await PSQLDatabase.close_pool()

    monkeypatch.setattr(dr, "keyword_search", _kw)

    def _bedrock_unreachable(*a, **k):
        raise RuntimeError("SYNTHETIC: Bedrock rerank unreachable in tests")

    monkeypatch.setattr(reranker, "_rerank_sync", _bedrock_unreachable)


@pytest.fixture()
def pool_cleanup():
    yield
    from app.services.database import PSQLDatabase

    if PSQLDatabase.pool is not None:
        PSQLDatabase.pool.terminate()
        PSQLDatabase.pool = None


def _store_with(monkeypatch, collection, fmts, path="default"):
    """A fresh real pgvector table holding the named SYNTHETIC fixtures, embedded through the
    REAL /embed route as userA, and a client wired to the chosen path."""
    store = _real_store(monkeypatch, collection)
    _add_keyword_column()
    client = _client(monkeypatch, store)
    monkeypatch.setattr("app.config.vector_store", store, raising=False)
    for fmt in fmts:
        fid, name, mime, make = FIXTURES[fmt]
        r = client.post("/embed", data={"file_id": fid, "entity_id": USER_A}, headers=_write_hdr(),
                        files={"file": (name, io.BytesIO(make()), mime)})
        assert r.status_code == 200, r.text
    _select_path(monkeypatch, path)
    return store, client


def _file_leg(client, file_id, query, hdr=None, **extra):
    body = {"file_id": file_id, "query": query, "k": CORE_K, **extra}
    return client.post("/query", json=body, headers=hdr or _core_hdr())


def _knowledge_leg(client, entity, query, hdr=None):
    body = {"query": query, "k": CORE_K, "args": {"cohere": False}}
    return client.post(f"/query/{entity}", json=body, headers=hdr or _core_hdr())


def _legs(client, file_id, query):
    """Both routes Core uses for this content, as (name, hits)."""
    out = []
    for name, resp in (("file_leg /query", _file_leg(client, file_id, query)),
                       ("knowledge_leg /query/{entity}", _knowledge_leg(client, USER_A, query))):
        assert resp.status_code == 200, (name, resp.text)
        out.append((name, resp.json()))
    return out


def _carrying(hits, token):
    return [h for h in hits if token in (h[0].get("page_content") or "")]


def _only(hits, token, where):
    found = _carrying(hits, token)
    assert len(found) == 1, f"{where}: expected exactly one chunk carrying {token!r}, got {len(found)}: {hits!r}"
    return found[0][0]["metadata"]


# =========================================================================== (A) ACCESS
@needs_pg
@pytest.mark.parametrize("path", ["default", "dense"])
def test_A_docx_block_index_reaches_both_core_routes_and_header_footer_carry_none_SYNTHETIC(
        monkeypatch, pool_cleanup, path):
    _, client = _store_with(monkeypatch, "c2p1_access_docx", ["docx"], path)
    for name, hits in _legs(client, F_DOCX, f"What is the {DOCX_TOKEN} quarterly revenue figure?"):
        md = _only(hits, DOCX_TOKEN, name)
        assert md.get("block_index") == 1, (name, md)
        assert (md.get("file_id"), md.get("user_id"), md.get("filename")) == (F_DOCX, USER_A, "report.docx"), md
        body_idx = sorted(h[0]["metadata"]["block_index"] for h in hits if "block_index" in h[0]["metadata"])
        assert body_idx == [0, 1, 2], (name, body_idx)
        # No fabrication on the READ path: header/footer units come back WITHOUT the key.
        for tok in (HDR_TOKEN, FTR_TOKEN):
            aux = _only(hits, tok, f"{name} aux {tok}")
            assert "block_index" not in aux, (name, tok, aux)
            assert aux.get("file_id") == F_DOCX, aux


@needs_pg
@pytest.mark.parametrize("path", ["default", "dense"])
def test_A_xlsx_cell_range_and_header_reach_both_core_routes_SYNTHETIC(monkeypatch, pool_cleanup, path):
    _, client = _store_with(monkeypatch, "c2p1_access_xlsx", ["xlsx"], path)
    for name, hits in _legs(client, F_XLSX, f"Which region carries {XLSX_TOKEN}?"):
        md = _only(hits, XLSX_TOKEN, name)
        assert md.get("cell_range") == "Revenue!A1:C3", (name, md)
        assert md.get("header") == ["Region", "Metric", "Value"], (name, md)
        assert md.get("header_row") == 1, (name, md)
        assert md.get("page_name") == "Revenue", (name, md)
        assert (md.get("file_id"), md.get("user_id")) == (F_XLSX, USER_A), md
        # The empty sheet has no occupied extent, emits no chunk, and so is NOT reachable
        # through the query path; its no-fabrication is pinned at the loader level
        # (test_xlsx_cell_locator::test_SYNTHETIC_empty_sheet_has_no_cell_range_at_the_locator_level).
        # What IS reachable: no chunk names the blank sheet, and every range present is
        # qualified to the chunk's OWN sheet.
        assert not [h for h in hits if h[0]["metadata"].get("page_name") == "Blank"], (name, hits)
        for h in hits:
            m = h[0]["metadata"]
            if "cell_range" in m:
                assert m["cell_range"].startswith(f"{m.get('page_name')}!"), (name, m)


@needs_pg
@pytest.mark.parametrize("path", ["default", "dense"])
def test_A_md_section_index_and_heading_path_reach_both_core_routes_preamble_carries_none_SYNTHETIC(
        monkeypatch, pool_cleanup, path):
    _, client = _store_with(monkeypatch, "c2p1_access_md", ["md"], path)
    for name, hits in _legs(client, F_MD, f"What does the {MD_TOKEN} revenue section say?"):
        md = _only(hits, MD_TOKEN, name)
        assert md.get("section_index") == 1, (name, md)
        assert md.get("heading_path") == "Overview > Revenue", (name, md)
        assert (md.get("file_id"), md.get("user_id"), md.get("filename")) == (F_MD, USER_A, "notes.md"), md
        pre = _only(hits, PRE_TOKEN, f"{name} preamble")
        assert "section_index" not in pre and "heading_path" not in pre, (name, pre)


# =========================================================================== (B) ISOLATION
@needs_pg
def test_B_a_file_scoped_query_never_returns_another_file_of_the_same_user_SYNTHETIC(
        monkeypatch, pool_cleanup):
    _, client = _store_with(monkeypatch, "c2p1_iso_file", ["docx", "md"])
    q = f"What does the {MD_TOKEN} revenue section say?"
    # Positive control: the md file DOES answer this question, so the negative is not vacuous.
    own = _file_leg(client, F_MD, q)
    assert own.status_code == 200 and _carrying(own.json(), MD_TOKEN), own.text
    other = _file_leg(client, F_DOCX, q)
    assert other.status_code == 200, other.text
    hits = other.json()
    assert hits, "precondition: the docx file has chunks of its own to return"
    assert {h[0]["metadata"].get("file_id") for h in hits} == {F_DOCX}, hits
    assert not _carrying(hits, MD_TOKEN), hits
    assert {h[0]["metadata"].get("user_id") for h in hits} == {USER_A}, hits


@needs_pg
def test_B_another_entity_retrieves_nothing_of_user_a_on_any_core_route_SYNTHETIC(
        monkeypatch, pool_cleanup):
    _, client = _store_with(monkeypatch, "c2p1_iso_entity", ["docx"])
    q = f"What is the {DOCX_TOKEN} quarterly revenue figure?"
    assert _carrying(_file_leg(client, F_DOCX, q).json(), DOCX_TOKEN), "positive control failed"
    hdr_b = _core_hdr((USER_B,))

    r = _file_leg(client, F_DOCX, q, hdr=hdr_b)            # file leg, A's file id, B's token
    assert r.status_code == 200, r.text
    assert r.json() == [], r.json()

    r = _knowledge_leg(client, USER_A, q, hdr=hdr_b)       # knowledge leg naming A
    assert r.status_code == 403, r.text
    assert not isinstance(r.json(), list), r.json()

    r = _knowledge_leg(client, USER_B, q, hdr=hdr_b)       # B's own knowledge leg
    assert r.status_code == 200, r.text
    assert [h for h in r.json() if h[0]["metadata"].get("user_id") != USER_B] == [], r.json()

    r = _file_leg(client, F_DOCX, q, hdr=hdr_b, entity_id=USER_A)   # agent-file shape naming A
    assert r.status_code == 403, r.text
    assert not isinstance(r.json(), list), r.json()


@needs_pg
def test_B_missing_or_invalid_auth_never_falls_back_to_data_SYNTHETIC(monkeypatch, pool_cleanup):
    _, client = _store_with(monkeypatch, "c2p1_iso_auth", ["docx"])
    q = f"What is the {DOCX_TOKEN} quarterly revenue figure?"
    assert _carrying(_file_leg(client, F_DOCX, q).json(), DOCX_TOKEN), "positive control failed"
    bad = {
        "missing": {"Content-Type": "application/json"},
        "not-bearer": {"Authorization": _token(), "Content-Type": "application/json"},
        "garbage": {"Authorization": "Bearer not.a.jwt", "Content-Type": "application/json"},
        "wrong-secret": {"Authorization": f"Bearer {_token(secret='SYNTHETIC-forged-secret')}",
                         "Content-Type": "application/json"},
        "expired": {"Authorization": f"Bearer {_token(hours=-1)}", "Content-Type": "application/json"},
    }
    for label, hdr in bad.items():
        for route, resp in (("/query", _file_leg(client, F_DOCX, q, hdr=hdr)),
                            ("/query/{entity}", _knowledge_leg(client, USER_A, q, hdr=hdr))):
            assert resp.status_code == 401, (label, route, resp.status_code, resp.text)
            assert not isinstance(resp.json(), list), (label, route, resp.text)
            assert DOCX_TOKEN not in resp.text, (label, route, resp.text)


# =========================================================================== (C) EXACT REPEAT
def _fresh_store(collection):
    """A NEW store object -- new SQLAlchemy engine, new connections -- bound to the SAME
    already-seeded table (no drop, no re-embed)."""
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store

    from app.services.vector_store.factory import get_vector_store

    store = get_vector_store(_sqlalchemy_dsn(), _DetEmb(), collection, mode="async")
    if store.create_extension:
        store.create_vector_extension()
    store.EmbeddingStore, store.CollectionStore = _get_embedding_collection_store(
        store._embedding_length, use_jsonb=store.use_jsonb)
    store.create_tables_if_not_exists()
    store.create_collection()
    return store


@needs_pg
@pytest.mark.parametrize("path", ["default", "dense"])
def test_C_same_question_twice_and_ten_times_and_after_a_fresh_app_is_byte_identical_SYNTHETIC(
        monkeypatch, pool_cleanup, path):
    from app.routes import document_routes as dr

    store, client = _store_with(monkeypatch, "c2p1_repeat", ["docx", "xlsx", "md"], path)
    q = "What is the quarterly revenue figure in euros?"
    calls = {
        "file_leg /query": lambda c: _file_leg(c, F_DOCX, q),
        "knowledge_leg /query/{entity}": lambda c: _knowledge_leg(c, USER_A, q),
    }
    baseline = {}
    for name, call in calls.items():
        r0 = call(client)
        assert r0.status_code == 200, (name, r0.text)
        hits = r0.json()
        assert len(hits) >= 3, f"{name}: precondition -- a multi-chunk ordered result, got {len(hits)}"
        assert all(isinstance(h[1], float) for h in hits), hits
        baseline[name] = r0.content
        r1 = call(client)                                   # asked twice
        assert r1.content == r0.content, f"{name}: the second identical request diverged"
        for i in range(10):                                 # asked N times
            ri = call(client)
            assert ri.content == r0.content, f"{name}: repeat {i + 2} diverged"

    # A fresh app instance: new store object, new engine and connections, new TestClient,
    # old engine disposed -- same table, nothing re-embedded.
    store._bind.dispose()
    monkeypatch.setattr(dr, "vector_store", _fresh_store("c2p1_repeat"))
    fresh = TestClient(client.app)
    for name, call in calls.items():
        rf = call(fresh)
        assert rf.status_code == 200, (name, rf.text)
        assert rf.content == baseline[name], f"{name}: a fresh app instance diverged from the baseline"

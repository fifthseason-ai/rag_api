"""DOCX per-unit locator fidelity (E1, SYNTHETIC fixtures only).

WHAT E1 ADDS. Until E1, DOCX flattened to ONE `locator_kind="none"` Document. E1's
`SafeDocxLoader` emits one Document per authored unit -- heading / paragraph /
table cell (row-major) / header / footer -- in `word/document.xml` body order,
each carrying the DOCX per-unit locator (a 0-based, contiguous block index), while
KEEPING the #85 AlternateContent single-emission (a text box is read once). One
additive family is registered in `_UNIT_LOCATOR_KEYS`.

This file pins that fidelity against the card's ground-truth shape and proves the
stamp reaches the STORED chunk, not only the loader Document (the FILES lead's
"a registered tuple entry that nothing stamps changes nothing" caution).

VALUE/KEY PENDING. The locator_kind vocabulary value and the cmetadata key are the
FILES lead's decision; nothing here hardcodes them. Every assertion reads
`SafeDocxLoader._DOCX_LOCATOR_KIND` / `._DOCX_LOCATOR_KEY`, so the ruling is a
one-token change. No literal of the value appears in this file (including names).

SYNTHETIC. Every fixture is hand-built OOXML generated at test time (docx2txt/ET
parse hand-rolled XML; LibreOffice would not open it). Synthetic proof establishes
loader behaviour on a KNOWN shape; it is NOT proof against a real client original,
which needs Graph/Box consent still outstanding (A3/A4).
"""

import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes
from app.routes.document_routes import (
    _UNIT_LOCATOR_KEYS,
    _extraction_receipt,
    _prepare_documents_sync,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import SafeDocxLoader, get_loader

# Reuse #85's synthetic OOXML builders rather than inventing new ones.
from tests.utils.test_docx_reading_order import DOCX_MIME, _TEXTBOX, _write_docx

# Read the family from the loader so the PENDING value/key ruling is a one-token
# change and no literal of the value is written anywhere in this file.
_KEY = SafeDocxLoader._DOCX_LOCATOR_KEY
_KIND = SafeDocxLoader._DOCX_LOCATOR_KIND

# DSN-gated real-Postgres round-trip (see test at the end of this file). Modelled on
# the existing gated suites (tests/services/test_pgvector_realpg.py,
# tests/utils/test_parse_is_not_index.py): RAG_TEST_PG_DSN selects the database and
# the port lives ONLY in that env var, so this file hardcodes no port and stays
# portable. With no DSN the round-trip test SKIPS cleanly (never fails), so a no-DB
# run stays deterministic; RAG_TEST_PG_REQUIRED=1 turns the skip into an error so a
# DB run can prove it actually executed.
PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the DOCX locator round-trip",
)

# Ground-truth tokens for the card fixture, in AUTHORED order. Distinct tokens so a
# scrambled or dropped unit is unambiguous.
_H1 = "H1_HEADING"
_P1 = "P1_BODY_BEFORE"
_C00, _C01, _C10, _C11 = "CELL_R0C0", "CELL_R0C1", "CELL_R1C0", "CELL_R1C1"
_H2 = "H2_HEADING"
_P2 = "P2_BODY_AFTER"
_TB = "ECHO_TEXTBOX pull quote"  # the exact run inside #85's _TEXTBOX

# The authored reading order of the card fixture. The four table cells are
# row-major (row 0 then row 1, left to right). The text box sits last.
_AUTHORED = [_H1, _P1, _C00, _C01, _C10, _C11, _H2, _P2, _TB]


def make_card_docx_SYNTHETIC(path):
    """The card's ground-truth shape: H1 -> P1 -> 2x2 table (row-major) -> H2 ->
    P2 -> text box (mc:AlternateContent, #85's _TEXTBOX). Built with #85's
    `_write_docx`/`_TEXTBOX`, so this is the same synthetic machinery, composed."""
    body = (
        f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>{_H1}</w:t></w:r></w:p>'
        f'<w:p><w:r><w:t>{_P1}</w:t></w:r></w:p>'
        "<w:tbl>"
        f'<w:tr><w:tc><w:p><w:r><w:t>{_C00}</w:t></w:r></w:p></w:tc>'
        f'<w:tc><w:p><w:r><w:t>{_C01}</w:t></w:r></w:p></w:tc></w:tr>'
        f'<w:tr><w:tc><w:p><w:r><w:t>{_C10}</w:t></w:r></w:p></w:tc>'
        f'<w:tc><w:p><w:r><w:t>{_C11}</w:t></w:r></w:p></w:tc></w:tr>'
        "</w:tbl>"
        f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>{_H2}</w:t></w:r></w:p>'
        f'<w:p><w:r><w:t>{_P2}</w:t></w:r></w:p>'
        f"{_TEXTBOX}"
    )
    _write_docx(path, body)


def _load_units(tmp_path):
    path = tmp_path / "card.docx"
    make_card_docx_SYNTHETIC(str(path))
    loader, known, ext = get_loader("card.docx", DOCX_MIME, str(path))
    assert known is True and ext == "docx"
    return str(path), loader.load()


def _which_unit(text, tokens=_AUTHORED):
    """The ground-truth token a unit's page_content carries (exactly one)."""
    hits = [t for t in tokens if t in text]
    assert len(hits) == 1, f"unit text {text!r} matched {hits!r}, expected exactly one"
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Every unit is indexed, in authored reading order, table cells row-major.
# ---------------------------------------------------------------------------


def test_every_unit_carries_the_family_index_in_authored_order_SYNTHETIC(tmp_path):
    path, docs = _load_units(tmp_path)

    # One unit per authored block, none dropped, none duplicated.
    assert [_which_unit(d.page_content) for d in docs] == _AUTHORED

    # Every unit carries the family key as an int; indices are 0-based, contiguous
    # and monotonic in reading order (so a citation resolves to a real position).
    assert all(isinstance(d.metadata.get(_KEY), int) for d in docs)
    assert [d.metadata[_KEY] for d in docs] == list(range(len(_AUTHORED)))

    # Provenance is always the uploaded file, never a working copy.
    assert all(d.metadata.get("source") == path for d in docs)


def test_table_cells_are_row_major_SYNTHETIC(tmp_path):
    _path, docs = _load_units(tmp_path)
    by_token = {_which_unit(d.page_content): d.metadata[_KEY] for d in docs}
    # row 0 before row 1; within a row, left before right.
    assert by_token[_C00] < by_token[_C01] < by_token[_C10] < by_token[_C11]
    # ...and contiguous, i.e. the four cells occupy four adjacent indices.
    assert sorted([by_token[_C00], by_token[_C01], by_token[_C10], by_token[_C11]]) == [
        by_token[_C00] + i for i in range(4)
    ]


def test_textbox_emitted_once_preserving_85_dedupe_SYNTHETIC(tmp_path):
    """#85 invariant preserved: an mc:AlternateContent text box carries the same
    runs in its Choice and Fallback; exactly one copy is emitted, and it sits after
    the table (last), never before it."""
    _path, docs = _load_units(tmp_path)
    joined = "\n".join(d.page_content for d in docs)
    assert joined.count(_TB) == 1, f"text box duplicated: {joined!r}"
    by_token = {_which_unit(d.page_content): d.metadata[_KEY] for d in docs}
    assert by_token[_C11] < by_token[_TB]  # after the table
    assert by_token[_TB] == len(_AUTHORED) - 1  # last


# ---------------------------------------------------------------------------
# 2. Receipt names the family AND agrees with the STORED chunks (not the loader
#    Document). Prepared in-process through _prepare_documents_sync.
# ---------------------------------------------------------------------------


def _kind_the_store_carries(stored_meta):
    """The family the STORED chunks carry, by the SAME registry + precedence the
    receipt uses. Copied deliberately from the general rule in
    test_receipt_locator_agrees_with_chunks.py so E1 proves agreement locally too."""
    for kind, key in _UNIT_LOCATOR_KEYS:
        if any((m or {}).get(key) is not None for m in stored_meta):
            return kind
    return "none"


def test_receipt_kind_is_the_family_and_agrees_with_prepared_chunks_SYNTHETIC(tmp_path):
    _path, docs = _load_units(tmp_path)
    prepared = _prepare_documents_sync(
        docs, "syn-file", "syn-user", True, "ORGANIC", "card.docx",
    )
    stored_meta = [dict(d.metadata or {}) for d in prepared]

    receipt = _extraction_receipt(docs)
    # The receipt names the DOCX family, not `none`.
    assert receipt["locator_kind"] == _KIND
    # The family the STORED chunks carry equals what the receipt claims.
    assert _kind_the_store_carries(stored_meta) == receipt["locator_kind"]
    # And every stored chunk actually carries the family key (the stamp survived
    # splitting + the service-field merge).
    assert all(isinstance(m.get(_KEY), int) for m in stored_meta)


# ---------------------------------------------------------------------------
# 3. The stamp reaches the STORE-INSERT boundary through the real /embed route.
#    A recording double stands in for AsyncPgVector.aadd_documents (no pgvector);
#    the DB round-trip itself is a pgvector run handed to the slot holder.
# ---------------------------------------------------------------------------


def _hdr(secret="testsecret", uid="testuser", tid="tenantA"):
    os.environ["JWT_SECRET"] = secret
    payload = {
        "id": uid, "tid": tid, "ent": ["userA"], "act": ["write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, secret, algorithm='HS256')}"}


def test_stamp_reaches_store_insert_boundary_through_embed_SYNTHETIC(tmp_path):
    """The strongest evidence available WITHOUT pgvector: the DOCX family reaches
    the arguments of `AsyncPgVector.aadd_documents` -- the last in-process point
    before the database insert -- and the /embed receipt names the family.

    LIMIT: this proves the stamp reaches the store-INSERT call, not that a row is
    written and read back. The DB round-trip is a pgvector-backed run (handed to
    the heavy-slot holder)."""
    os.environ["JWT_SECRET"] = "testsecret"
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    path = tmp_path / "card.docx"
    make_card_docx_SYNTHETIC(str(path))

    added = []

    async def recording_aadd(self, docs, ids=None, executor=None):
        added.append(list(docs))
        return ids

    async def dummy_delete(self, ids=None, collection_only=False, user_id=None,
                           document_origin_type=None, subscription_id=None, executor=None, **_):
        return None

    orig_add = AsyncPgVector.aadd_documents
    orig_del = AsyncPgVector.delete
    AsyncPgVector.aadd_documents = recording_aadd
    AsyncPgVector.delete = dummy_delete
    try:
        client = TestClient(app)
        r = client.post(
            "/embed",
            data={"file_id": "f-docx", "entity_id": "userA"},
            files={"file": ("card.docx", io.BytesIO(path.read_bytes()), DOCX_MIME)},
            headers=_hdr(),
        )
        assert r.status_code == 200, r.text
        receipt = r.json()["extraction"]
        stored_meta = [dict(d.metadata or {}) for batch in added for d in batch]
    finally:
        AsyncPgVector.aadd_documents = orig_add
        AsyncPgVector.delete = orig_del

    assert stored_meta, "no chunks reached the store-insert boundary"
    # The receipt names the family and the inserted chunks carry it -- they agree.
    assert receipt["locator_kind"] == _KIND
    assert _kind_the_store_carries(stored_meta) == _KIND
    assert all(isinstance(m.get(_KEY), int) for m in stored_meta)
    # The text box was inserted once, not twice (#85 preserved end to end).
    joined = "\n".join(
        getattr(d, "page_content", "") or "" for batch in added for d in batch
    )
    assert joined.count(_TB) == 1, f"text box duplicated in the store: {joined!r}"


# ---------------------------------------------------------------------------
# 4. The registry tuple line mirrors the loader constants (the routes comment
#    promises this; here it is pinned so the two cannot drift).
# ---------------------------------------------------------------------------


def test_registry_entry_mirrors_loader_constants():
    assert (_KIND, _KEY) in _UNIT_LOCATOR_KEYS, (
        "the DOCX tuple in _UNIT_LOCATOR_KEYS must equal "
        "(SafeDocxLoader._DOCX_LOCATOR_KIND, ._DOCX_LOCATOR_KEY); it drifted"
    )
    # Appended at the END: the detection loop breaks on the first family present,
    # and DOCX carries exactly this family, so placement is order-safe.
    assert _UNIT_LOCATOR_KEYS[-1] == (_KIND, _KEY)


# ---------------------------------------------------------------------------
# 5. Controls -- prove the checks above can fail (no vacuous green).
# ---------------------------------------------------------------------------


def test_family_check_fails_when_the_stamp_is_missing():
    """Positive control for the stamp evidence: `_kind_the_store_carries` must NOT
    report the family for a chunk that carries no locator key, and the int check
    must reject a missing/None index. If either passed here, tests 2-3 would be
    vacuous."""
    assert _kind_the_store_carries([{"source": "x"}]) == "none"
    assert _kind_the_store_carries([{"source": "x", _KEY: None}]) == "none"
    assert not isinstance(({"source": "x"}).get(_KEY), int)
    # ...and it DOES recover the family from a chunk carrying only the family key.
    assert _kind_the_store_carries([{_KEY: 0}]) == _KIND


def test_the_fixture_actually_builds_the_nine_ground_truth_tokens(tmp_path):
    """Guards the premise: if the fixture stopped producing a token, every order
    assertion about it would be vacuously satisfiable. Assert the raw XML carries
    all nine authored tokens before any loader runs."""
    import zipfile

    path = str(tmp_path / "card.docx")
    make_card_docx_SYNTHETIC(path)
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    for token in _AUTHORED:
        assert token in xml, f"fixture no longer emits {token!r}"


# ---------------------------------------------------------------------------
# 6. REAL pgvector round-trip (DSN-gated). The fast tests above prove the stamp
#    reaches the last IN-PROCESS point (the recording double on aadd_documents).
#    INTEGRATION requires the stamp be proven on a REAL DB round-trip before the PR:
#    this drives the real /embed route into a real Postgres+pgvector and reads the
#    values back OUT of the table through the store's OWN product read path. Runs
#    only when RAG_TEST_PG_DSN is set; skips cleanly otherwise (adds exactly ONE
#    skip to a no-DB run). SYNTHETIC fixture only.
# ---------------------------------------------------------------------------


def _real_store(collection):
    """A vector store bound to the real Postgres named by RAG_TEST_PG_DSN, with the
    pgvector tables dropped and recreated so counts are deterministic. Mirrors the
    proven setup in test_parse_is_not_index.py::_real_store (conftest no-ops the
    pgvector __post_init__ for the session, so it is run explicitly here)."""
    import psycopg2
    from app.services.vector_store.factory import get_vector_store
    from tests.utils.test_empty_entitlement_query_path import _DetEmb

    raw = PG_DSN.replace("postgresql+psycopg2://", "postgresql://")
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE"
        )
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


@needs_pg
def test_docx_block_locator_round_trips_through_real_pgvector_SYNTHETIC(monkeypatch, tmp_path):
    """The DOCX block locator survives a REAL Postgres round-trip: every persisted row,
    read BACK OUT of the table, carries the family index in 0-based contiguous authored
    order; the /embed receipt names the family and AGREES with the rows; no foreign
    locator key rides along; and the uploaded filename is preserved.

    READ-BACK PATH. Rows are read through the store's own `get_documents_by_ids` -- the
    product read path GET /documents uses -- so the real read boundary is exercised, not
    a hand-rolled SELECT. Raw SQL (psycopg2) is used ONLY to DROP/reset the tables in
    setup, never to make an assertion; `cmetadata` comes back from Postgres via the store.

    ABLE TO FAIL. The single targeted mutation that reddens the fast in-process test --
    deleting the `self._DOCX_LOCATOR_KEY: block_index` stamp line in SafeDocxLoader.load()
    -- also reddens THIS test: the loader still emits one Document per block, so rows are
    still stored and read back, but each row's cmetadata carries no `_KEY`, and assertion
    (1) fires first with a message naming the missing stored value ("the loader stamp did
    not survive to the Postgres row"). Assertion (2) would fire too (the receipt would say
    'none', not the family).

    LIMIT. Even with the round-trip proven, this is a SYNTHETIC OOXML fixture; it proves
    loader->route->pgvector fidelity on a KNOWN shape, not against a real client original
    (Graph/Box consent still outstanding, A3/A4). Embeddings are the offline `_DetEmb`, so
    vector *values* are not exercised -- only the metadata/text a citation reads.
    """
    from app.services.vector_store.extended_pg_vector import ExtendedPgVector

    UPLOAD_NAME = "card.docx"
    FID = "e1-docx-roundtrip"

    store = _real_store("e1_docx_roundtrip")
    os.environ["JWT_SECRET"] = "testsecret"
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(document_routes, "vector_store", store)

    path = tmp_path / UPLOAD_NAME
    make_card_docx_SYNTHETIC(str(path))

    client = TestClient(app)
    r = client.post(
        "/embed",
        data={"file_id": FID, "entity_id": "userA"},
        files={"file": (UPLOAD_NAME, io.BytesIO(path.read_bytes()), DOCX_MIME)},
        headers=_hdr(),
    )
    assert r.status_code == 200, r.text
    receipt = r.json()["extraction"]

    # Read the rows back OUT of Postgres through the store's own product read path.
    rows = ExtendedPgVector.get_documents_by_ids(store, [FID])
    metas = [dict(d.metadata or {}) for d in rows]

    # Precondition: rows really landed and were read back (so nothing below is vacuous).
    assert metas, "no DOCX chunks were read back from Postgres"
    assert len(metas) == len(_AUTHORED), (
        f"expected {len(_AUTHORED)} block-indexed chunks in the table, read back "
        f"{len(metas)}: {[m.get(_KEY) for m in metas]}"
    )

    # (1) EXACT per-row locator value, read from the DB, against the AUTHORED order.
    #     Each row's block index must equal its ground-truth token's position in
    #     _AUTHORED (0-based, reading order); the set must be 0..N-1 contiguous.
    seen = {}
    for d, m in zip(rows, metas):
        idx = m.get(_KEY)
        assert isinstance(idx, int), (
            f"stored chunk carries no int {_KEY!r} (value {idx!r}): the loader stamp did "
            f"not survive to the Postgres row -- row cmetadata read back: {m}"
        )
        token = _which_unit(d.page_content)
        assert idx == _AUTHORED.index(token), (
            f"stored block index {idx} != authored position {_AUTHORED.index(token)} for "
            f"unit {token!r} (rows out of order or misindexed in the table)"
        )
        seen[idx] = token
    assert sorted(seen) == list(range(len(_AUTHORED))), (
        f"block indices read back are not 0-based contiguous: {sorted(seen)}"
    )

    # (2) The /embed receipt names the family AND agrees with what the rows carry: the
    #     receipt must not be able to claim a position the store cannot support.
    assert receipt["locator_kind"] == _KIND, receipt
    assert _kind_the_store_carries(metas) == receipt["locator_kind"], (
        f"receipt says locator_kind={receipt['locator_kind']!r} but the stored rows carry "
        f"{_kind_the_store_carries(metas)!r} (the F-DOCX1/F-NONE1 disagreement)"
    )

    # (3) EXCLUSION: a DOCX row carries NO OTHER registered locator key. Assert what is
    #     EXCLUDED (page / slide_number / page_name / row), not only what is contained.
    other_keys = [key for _k, key in _UNIT_LOCATOR_KEYS if key != _KEY]
    for m in metas:
        foreign = [k for k in other_keys if k in m]
        assert not foreign, (
            f"a stored DOCX row carries foreign locator key(s) {foreign}: {m}"
        )

    # (4) PROVENANCE: the uploaded filename is preserved on every stored row, and the
    #     server-side `source` path derives from it (no other file's bytes leaked in).
    #     Through /embed, `source` is the route's unique temp path (loader-set,
    #     `<stem>_<hex>.docx`); the user-facing uploaded name is the `filename` field.
    for m in metas:
        assert m.get("filename") == UPLOAD_NAME, m
        src = m.get("source")
        assert (
            isinstance(src, str)
            and os.path.basename(src).startswith("card_")
            and src.endswith(".docx")
        ), f"source not derived from the uploaded file: {src!r}"

    # #85 dedupe preserved end to end: the text box is stored exactly once in the DB.
    joined = "\n".join(d.page_content for d in rows)
    assert joined.count(_TB) == 1, f"text box duplicated in Postgres: {joined!r}"

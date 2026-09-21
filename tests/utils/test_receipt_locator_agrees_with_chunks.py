"""The receipt's `locator_kind` must name the family the stored chunks actually carry — for EVERY format.

FILES-DEV F-RECEIPT-AGREE. Generalises the CSV-only finding of #47 (F-NONE1) and the DOCX-only
finding of #43 (F-DOCX1) into one rule checked across all seven intake formats.

THE RULE (two halves of one contract):
  1. If the stored chunks carry a registered locator key, the receipt's `locator_kind` must name
     that family.  (A receipt that says `none` while the store carries `page`/`row`/... is the
     F-NONE1 / F-DOCX1 disagreement: a consumer reading chunk metadata can build a position the
     receipt denies.)
  2. `none` must mean no stored chunk carries ANY registered locator key.

Why this is the same defect twice, generalised:
  * #43 measured that a `page=0` stamped only in `_prepare_documents_sync` (the STORE) — never in
    the loader the receipt reads — was invisible to the whole existing suite: the receipt said
    `none` while every persisted DOCX chunk carried `page`.  That is the store and the receipt
    disagreeing, and it is exactly what `test_receipt_locator_kind_names_the_family_the_store_carries`
    catches, for every format, because it recomputes the family FROM THE STORED CHUNKS and compares
    it to what the receipt claimed.
  * #47 measured that on `main` a CSV chunk carries `row` while the receipt says `none`, because
    `row` was not yet registered.  #36 closes that by registering `("row", "row")`.  This file is
    stacked on #36, so on this tree the rule HOLDS for CSV and is proven green rather than xfail.

DRIVEN FROM `_UNIT_LOCATOR_KEYS`, never a copied list:  Test 1 reads the registry itself to decide
which family the stored chunks carry, so a locator family added later is checked against all seven
formats automatically.

THE INDEPENDENT ANCHOR, and why it must exist:  Test 2 checks that the registry is EXACTLY the set
of locator keys the real loaders emit.  It cannot do that by reading `_UNIT_LOCATOR_KEYS` on both
sides — a set compared against itself is a change-detector that can never fail (the standing lesson
"a control indistinguishable from its absence").  So the keys the loaders actually emit are pinned
here, `_LOADER_NATIVE_LOCATOR_KEYS`, MEASURED through the real /embed route (see the probe in the
PR body), independent of the module.  When someone edits the registry, the module and this anchor
diverge, and Test 2 reddens — which is the whole point.

Store double:  every fixture is driven through the REAL /embed route; `AsyncPgVector.aadd_documents`
is recorded so the assertions see the ACTUAL stored chunk metadata, not the loader's pre-persist
metadata.  Modelled on tests/utils/test_extraction_status.py::rec_client.
"""

import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes.document_routes import _UNIT_LOCATOR_KEYS
from app.services.vector_store.async_pg_vector import AsyncPgVector

# Real fixture generators already proven in the suite (real loaders, real bytes).
from tests.utils.test_parser_fitness import make_docx, make_multisheet_xlsx
from tests.utils.test_extraction_status import (
    make_partial_pdf,
    make_partial_pptx,
    make_markdown,
)

_SECRET = "testsecret"
_MARKER = "RECEIPT-AGREE-MARKER-7731"

# Every intake format the locator contract covers. TXT and CSV are built inline; the rest reuse
# the suite's proven generators.
_FORMATS = ["pdf", "pptx", "xlsx", "docx", "md", "txt", "csv"]

# INDEPENDENT GROUND TRUTH — the locator metadata key each loader emits NATIVELY, measured through
# the real /embed route, NOT read from _UNIT_LOCATOR_KEYS. Keyed by metadata key -> the format that
# produces it. This is the anchor Test 2 compares the registry against; see the module docstring for
# why it must be independent of the module.
_LOADER_NATIVE_LOCATOR_KEYS = {
    "page": "pdf",           # SafePyPDFLoader / pypdf
    "slide_number": "pptx",  # SlidePowerPointLoader
    "page_name": "xlsx",     # UnstructuredExcelLoader mode="elements"
    "row": "csv",            # RowCSVLoader / langchain CSVLoader (registered by #36)
}
# Formats that carry no per-unit locator at all -> must fold to a single `none` unit.
_NONE_FORMATS = {"docx", "md", "txt"}


# ---------------------------------------------------------------------------
# Store double + real /embed driver (modelled on test_extraction_status.rec_client)
# ---------------------------------------------------------------------------


def _hdr(uid="testuser", tid="tenantA"):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid, "tid": tid, "ent": ["userA"], "act": ["write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _build_fixtures(tmp_path):
    """Write one real file per format. Every text-bearing unit carries _MARKER so a loader that
    silently produced empty documents cannot satisfy the locator rule for the wrong reason."""
    paths = {}

    pdf = tmp_path / "doc.pdf"
    make_partial_pdf(str(pdf))  # pages 0,1,2 carry native text; 3,4 blank -> locator `page`
    paths["pdf"] = (pdf, "application/pdf")

    pptx = tmp_path / "doc.pptx"
    make_partial_pptx(str(pptx))  # slide 1 text (+notes/table), slide 2 image-only -> `slide`
    paths["pptx"] = (
        pptx, "application/vnd.openxmlformats-officedocument.presentationml.presentation")

    xlsx = tmp_path / "doc.xlsx"
    make_multisheet_xlsx(str(xlsx))  # two data sheets -> `sheet`
    paths["xlsx"] = (
        xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    docx = tmp_path / "doc.docx"
    make_docx(str(docx))  # no per-unit locator -> `none`
    paths["docx"] = (
        docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    md = tmp_path / "doc.md"
    md.write_text("# Heading\n\n%s in a markdown body.\n\n## Second\n\nMore.\n" % _MARKER,
                  encoding="utf-8")
    paths["md"] = (md, "text/markdown")

    txt = tmp_path / "doc.txt"
    txt.write_text("%s in a plain text body, no locator.\n" % _MARKER, encoding="utf-8")
    paths["txt"] = (txt, "text/plain")

    csv = tmp_path / "doc.csv"
    csv.write_text("region,revenue\nEMEA,%s\nAMER,3100000\n" % _MARKER, encoding="utf-8")
    paths["csv"] = (csv, "text/csv")

    return paths


@pytest.fixture(scope="module")
def format_matrix(tmp_path_factory):
    """Drive every format through the REAL /embed route once, capturing per format the receipt
    (from the response body) and the ACTUAL stored chunk metadata (from the recorded inserts).

    Module-scoped and self-monkeypatched (save -> setattr -> yield -> restore) so the seven embeds
    run once rather than once per parametrized assertion."""
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

    orig_add = AsyncPgVector.aadd_documents
    orig_del = AsyncPgVector.delete
    AsyncPgVector.aadd_documents = recording_aadd
    AsyncPgVector.delete = dummy_delete
    try:
        client = TestClient(app)
        tmp_path = tmp_path_factory.mktemp("receipt_agree")
        paths = _build_fixtures(tmp_path)
        matrix = {}
        for fmt in _FORMATS:
            path, ctype = paths[fmt]
            added.clear()
            r = client.post(
                "/embed",
                data={"file_id": "f-%s" % fmt, "entity_id": "userA"},
                files={"file": (path.name, io.BytesIO(path.read_bytes()), ctype)},
                headers=_hdr(),
            )
            assert r.status_code == 200, "%s: /embed returned %s: %s" % (
                fmt, r.status_code, r.text)
            receipt = r.json()["extraction"]
            stored_meta = [dict(d.metadata or {}) for batch in added for d in batch]
            stored_text = "\n".join(
                getattr(d, "page_content", "") or "" for batch in added for d in batch)
            matrix[fmt] = {"receipt": receipt, "stored": stored_meta, "text": stored_text}
    finally:
        AsyncPgVector.aadd_documents = orig_add
        AsyncPgVector.delete = orig_del
    return matrix


# ---------------------------------------------------------------------------
# Preconditions — without these the rule can be true of a pipeline that stored nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", _FORMATS)
def test_every_format_actually_stored_chunks(format_matrix, fmt):
    assert format_matrix[fmt]["stored"], (
        "%s stored no chunks, so every locator assertion about it is vacuous" % fmt)


# A known text token each fixture is guaranteed to carry through extraction. md/txt/csv are written
# in this file with _MARKER; pdf/docx carry their generator's own known strings; pptx/xlsx reflow
# cell/shape/notes text, so a specific known cell value is used instead of a raw marker.
_CONTENT_TOKEN = {
    "pdf": "native page",             # make_partial_pdf writes "... native page N revenue"
    "pptx": "EUR 4.2M",               # make_partial_pptx table cell
    "xlsx": "Widget",                 # make_multisheet_xlsx cell A2 on sheet "Sales"
    "docx": "Executive Summary",      # make_docx heading
    "md": _MARKER,
    "txt": _MARKER,
    "csv": _MARKER,
}


@pytest.mark.parametrize("fmt", _FORMATS)
def test_content_survived_extraction(format_matrix, fmt):
    """A loader that silently produced empty documents would satisfy the locator rule perfectly and
    for the wrong reason. Each format is checked for a token it is known to carry through."""
    assert _CONTENT_TOKEN[fmt] in format_matrix[fmt]["text"], (
        "%s: expected content token %r did not survive extraction"
        % (fmt, _CONTENT_TOKEN[fmt]))


# ---------------------------------------------------------------------------
# TEST 1 — THE GENERAL RULE, driven from _UNIT_LOCATOR_KEYS, across all seven formats.
# ---------------------------------------------------------------------------


def _kind_the_store_carries(stored_meta):
    """The locator family the STORED chunks carry, decided by _UNIT_LOCATOR_KEYS precedence — the
    same registry and same precedence _extraction_receipt uses, but read off the persisted chunks
    rather than the loader documents the receipt saw. When the two disagree, the receipt is claiming
    (or denying) a position the store does not (or does) carry."""
    for kind, key in _UNIT_LOCATOR_KEYS:
        if any((md or {}).get(key) is not None for md in stored_meta):
            return kind
    return "none"


@pytest.mark.parametrize("fmt", _FORMATS)
def test_receipt_locator_kind_names_the_family_the_store_carries(format_matrix, fmt):
    """THE RULE. For every format, the receipt's locator_kind must name the family the stored chunks
    actually carry, and `none` must mean the store carries no registered locator key."""
    receipt = format_matrix[fmt]["receipt"]
    stored = format_matrix[fmt]["stored"]
    expected = _kind_the_store_carries(stored)
    assert receipt["locator_kind"] == expected, (
        "%s: receipt says locator_kind=%r but the stored chunks carry %r "
        "(per _UNIT_LOCATOR_KEYS %r). The receipt and the store disagree about the citable "
        "position." % (fmt, receipt["locator_kind"], expected,
                        [k for _, k in _UNIT_LOCATOR_KEYS]))


# ---------------------------------------------------------------------------
# TEST 2 — REGISTRY SOUNDNESS. The registry must be EXACTLY the locator keys the loaders emit.
# The two card controls target this test, and they redden DIFFERENT assertions.
# ---------------------------------------------------------------------------


def test_registry_is_exactly_the_locator_keys_the_loaders_emit(format_matrix):
    registered = {key for _kind, key in _UNIT_LOCATOR_KEYS}

    # What the loaders ACTUALLY emit in this matrix, measured against the independent anchor.
    observed = set()
    for fmt in _FORMATS:
        stored = format_matrix[fmt]["stored"]
        for key in _LOADER_NATIVE_LOCATOR_KEYS:
            if any((md or {}).get(key) is not None for md in stored):
                observed.add(key)

    # Fixture guard: the matrix must still exercise every native key we anchored. If a loader stops
    # emitting one (e.g. `row` disappears after being registered), this reddens naming the loader,
    # not the registry — a different defect from the two below.
    assert observed == set(_LOADER_NATIVE_LOCATOR_KEYS), (
        "the fixtures no longer emit the anchored native locator keys: expected %r, observed %r "
        "(a loader stopped emitting a locator, or a fixture stopped producing that format)"
        % (sorted(_LOADER_NATIVE_LOCATOR_KEYS), sorted(observed)))

    # CONTROL A — "register a key nothing emits" reddens HERE: a registered family with no producing
    # format means locator_kind could name a position no loader ever stores.
    assert registered <= observed, (
        "_UNIT_LOCATOR_KEYS registers %r that NO format in this matrix emits — either add a fixture "
        "that produces it, or the receipt can claim a locator family the store never carries"
        % sorted(registered - observed))

    # CONTROL B — "unregister one that is emitted" reddens HERE: a loader emits a locator the
    # registry ignores, so the receipt reports locator_kind=none while the chunks carry a position.
    assert observed <= registered, (
        "loaders emit locator keys %r that _UNIT_LOCATOR_KEYS does not register — the receipt will "
        "say locator_kind=none while the stored chunks carry that position (the F-NONE1 split)"
        % sorted(observed - registered))


# ---------------------------------------------------------------------------
# TEST 3 — guard the guard. If _kind_the_store_carries ignored the metadata, Test 1 would pass
# against any implementation at all.
# ---------------------------------------------------------------------------


def test_kind_the_store_carries_actually_reads_the_metadata():
    assert _UNIT_LOCATOR_KEYS, "_UNIT_LOCATOR_KEYS is empty; this whole file is vacuous"
    # Every registered family must be recovered from a chunk that carries only its key.
    for kind, key in _UNIT_LOCATOR_KEYS:
        assert _kind_the_store_carries([{key: 0}]) == kind, (
            "the store-side recompute did not recover family %r from key %r" % (kind, key))
    # A value of None is not a locator (matches _extraction_receipt's `is not None` test).
    first_key = _UNIT_LOCATOR_KEYS[0][1]
    assert _kind_the_store_carries([{first_key: None}]) == "none"
    # No registered key -> none.
    assert _kind_the_store_carries([{"not_a_locator": 1}]) == "none"
    assert _kind_the_store_carries([]) == "none"

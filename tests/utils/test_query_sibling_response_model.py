"""The two sibling query routes now publish the SAME response shape `/query` does -- and
this test proves the shape is identical where they agree, and pins the one place they DON'T.

BACKGROUND. #45 (F-QC1) gave `/query` a `response_model=List[QueryHit]` describing what the
route already returns, proven byte-identical on the wire. `/query/{entity_id}` and
`/query_multiple` return the related shape and did NOT get one. The card's first duty was to
establish -- by MEASUREMENT, not assumption -- whether the siblings return the SAME shape.

MEASURED (throwaway probe, raw response bytes, store double, five-format metadata union):

  NON-EMPTY  all three routes return byte-for-byte:
             [[{"id":null,"metadata":{<open dict>},"page_content":...,"type":"Document"}, <float>]]
             -> the asymmetry (only `/query` documented) was a real gap, closed additively here.

  EMPTY (originally)  `/query`             -> []                                    (200)
                      `/query/{entity_id}` -> []                                    (200)
                      `/query_multiple`    -> {"detail":"No documents found ..."}   (404)
             -> a REAL, undocumented divergence #52 first pinned: `/query_multiple` signalled
                "no results" as an ERROR a consumer had to catch, while its two siblings signalled
                it as an empty list.

  EMPTY (now, this PR)  ALL THREE -> [] (200).
             -> The divergence is REMOVED, not merely re-pinned. Core has ZERO call sites of
                `/query_multiple` (`git grep query_multiple` == 0 at Core 881a124cf and its
                Candidate B composition), and asked Files to align empty -> []/200 in a separate
                explicit PR -- CORE-TO-FILES-CONTRACT-ANSWERS-20260921.md (Q3). Only the empty
                branch changed; the non-empty wire is byte-identical (proven below and by probe).

WHY `metadata` STAYS AN OPEN DICT. FastAPI validates and re-serialises through the model, so
naming metadata's keys would DELETE every locator a citation is built from -- page, page_label,
page_name, page_number, slide_number, slide_title, row -- while the diff reads as additive. That
silent-deletion failure is what the metadata-survival test below makes impossible, on BOTH
siblings, the same guard #45 put on `/query`.
"""

import datetime
import os

import jwt
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# Must be set before importing main (app.config refuses to import without it).
os.environ.setdefault("JWT_SECRET", "testsecret")

from app.routes import document_routes  # noqa: E402
from main import app  # noqa: E402

#: The five-format metadata union #45 measured, plus provenance. No field here is named by the
#: model; every one must survive re-serialisation. Chosen to look like a realistic parser upgrade,
#: not obvious junk -- the likely cause of a future narrowing is a new field, not inserted nonsense.
UNMODELLED = {
    "page_name": "Cost Detail", "page_number": 2,        # spreadsheet
    "slide_number": 5, "slide_title": "Roadmap",         # presentation
    "page": 0, "page_label": "iii", "total_pages": 9,     # pdf
    "row": 7,                                             # csv
    "text_source": "ocr",
    "a_key_invented_after_this_model_was_written": "must survive",
    "creator": "PyPDF", "producer": "pypdf", "category": "Table",
    "filetype": "application/pdf", "file_directory": "/tmp/uploads/userA",
    "creationdate": "", "languages": ["eng"],
    "text_as_html": "<table><tr><td>a</td></tr></table>",
}

# The three routes, each with a request body that reaches the shared producer. `/query/{entity_id}`
# takes its filter from the PATH; `/query` and `/query_multiple` from the body. The entity in every
# case is userA so the route's own defensive filter (where present) keeps the hit.
ROUTES = {
    "/query": ("/query", {"query": "q", "file_id": "f1", "k": 1, "entity_id": "userA"}),
    "/query/{entity_id}": ("/query/userA", {"query": "q", "k": 1}),
    "/query_multiple": ("/query_multiple", {"query": "q", "file_ids": ["f1"], "k": 1}),
}


def _auth():
    """The entity MUST match the hit's user_id, or the route filters it out before the model
    is ever reached (`/query` and `/query_multiple` re-filter defensively)."""
    secret = os.environ["JWT_SECRET"]
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"],
        "act": ["read", "write", "delete"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, secret, algorithm="HS256")}


def _install(monkeypatch, retrieve_result):
    async def fake_retrieve(*_args, **_kwargs):
        return retrieve_result

    monkeypatch.setattr(document_routes, "_retrieve_documents", fake_retrieve)
    monkeypatch.setattr(document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3])
    # NOT a context manager: entering the lifespan opens a real Postgres connection the shape
    # test neither has nor needs. The suite's own client initialises the pool by hand.
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
    return TestClient(app)


@pytest.fixture()
def one_hit_client(monkeypatch):
    meta = {"file_id": "f1", "user_id": "userA", "tenant_id": "tenantA"}
    meta.update(UNMODELLED)
    return _install(monkeypatch, [(Document(page_content="the passage", metadata=meta), 0.25)])


@pytest.fixture()
def empty_client(monkeypatch):
    return _install(monkeypatch, [])


def _post(client, label):
    path, body = ROUTES[label]
    return client.post(path, json=body, headers=_auth())


# --- The shape the siblings SHARE with `/query`, on the non-empty path -------------------------

@pytest.mark.parametrize("label", list(ROUTES))
def test_every_metadata_key_survives_on_every_route(one_hit_client, label):
    """The one that matters. A narrowed model deletes locators and looks additive doing it --
    on the siblings exactly as on `/query`. The failure control for this is a narrowed
    QueryDocument.metadata: it reddens this assertion on all three routes (see receipt)."""
    r = _post(one_hit_client, label)
    assert r.status_code == 200, r.text
    hit = r.json()[0]
    doc = hit[0] if isinstance(hit, list) else hit
    meta = doc["metadata"]

    missing = {k: v for k, v in UNMODELLED.items() if k not in meta}
    assert not missing, (
        "%s: the response model DELETED metadata keys: %s. `metadata` must stay an open dict -- "
        "typing it removes every locator a citation is built from." % (label, sorted(missing))
    )
    for key, value in UNMODELLED.items():
        assert meta[key] == value, (
            "%s: %s came back as %r, not %r -- the model coerced a value it should have passed "
            "through" % (label, key, meta[key], value)
        )


@pytest.mark.parametrize("label", list(ROUTES))
def test_envelope_is_a_pair_not_an_object(one_hit_client, label):
    """Existing wire: `[[document, score], ...]`. A model may describe it, not reshape it --
    every consumer indexes position 0 and 1 today."""
    hit = _post(one_hit_client, label).json()[0]
    assert isinstance(hit, list) and len(hit) == 2, "%s: envelope changed shape: %r" % (label, hit)
    assert isinstance(hit[1], (int, float)), "%s: score is no longer a number: %r" % (label, hit[1])


@pytest.mark.parametrize("label", list(ROUTES))
def test_document_still_carries_id_and_type(one_hit_client, label):
    """LangChain serialises both; omitting them from the model would drop them from the wire."""
    doc = _post(one_hit_client, label).json()[0][0]
    for key in ("id", "type", "page_content", "metadata"):
        assert key in doc, "%s: the model dropped %r from the document: %s" % (label, key, sorted(doc))


def test_all_three_routes_return_the_identical_non_empty_body(one_hit_client, monkeypatch):
    """Source-of-truth equality: three consumers of ONE producer (`_retrieve_documents`) must put
    the SAME bytes on the wire for the same hit. A fresh client per route (the fixture is
    function-scoped) so state cannot leak between them.

    The hit carries a DECLARED score kind (F-QUERY-SCORE-KIND-ON-THE-WIRE; RV-118 note 2): with a
    plain-list stub every route sends score_kind null and the comparison could not see a route
    that dropped or changed the kind. Declared, the equality below also proves all three routes
    put the same kind on the wire -- and the last assert proves the kind is IN the compared bytes."""
    from app.services.score_kind import RRF, ScoredHits
    bodies = {}
    for label in ROUTES:
        client = _install(monkeypatch, ScoredHits(
            [(Document(page_content="the passage",
                       metadata={**{"file_id": "f1", "user_id": "userA", "tenant_id": "tenantA"},
                                 **UNMODELLED}), 0.25)], RRF))
        bodies[label] = _post(client, label).content
    distinct = set(bodies.values())
    assert len(distinct) == 1, (
        "the sibling routes disagree on the non-empty wire: %s" %
        {k: v.decode("utf-8", "replace") for k, v in bodies.items()}
    )
    assert b'"score_kind":"rrf"' in distinct.pop(), "the declared kind never reached the compared bytes"


# --- The empty result: all three routes now AGREE on [] 200 -----------------------------------

def test_query_empty_result_is_an_empty_list_200(empty_client):
    r = _post(empty_client, "/query")
    assert r.status_code == 200, r.text
    assert r.json() == [], r.text


def test_query_by_entity_empty_result_is_an_empty_list_200(empty_client):
    r = _post(empty_client, "/query/{entity_id}")
    assert r.status_code == 200, r.text
    assert r.json() == [], r.text


def test_query_multiple_empty_result_is_now_an_empty_list_200(empty_client):
    """THE PRODUCT CHANGE THIS PR MAKES, pinned on the wire. `/query_multiple` previously
    answered 404 {"detail":"No documents found for the given query"} on no results; #52 pinned
    that divergence. This PR aligns it to `[]` 200 to match its two siblings (Core Q3 decision,
    CORE-TO-FILES-CONTRACT-ANSWERS-20260921.md -- Core has zero call sites of this route). The
    failure control reverts the route's empty branch to the 404 raise and this reddens on both
    the status assertion (404 != 200) and the body assertion (the 404 detail object != [])."""
    r = _post(empty_client, "/query_multiple")
    assert r.status_code == 200, r.text
    assert r.json() == [], r.text
    # The former 404 detail object must no longer appear anywhere in the empty response.
    assert r.text.strip() == "[]", r.text


def test_all_three_query_routes_agree_on_empty_result(empty_client, monkeypatch):
    """Source-of-truth equality on the EMPTY path, the mirror of the non-empty equality test.
    Three consumers of one producer, given nothing, must put the SAME bytes on the wire. This is
    the invariant the alignment establishes; if any route regresses (e.g. the 404 returns to
    `/query_multiple`), the set of distinct bodies grows and this reddens."""
    bodies = {}
    for label in ROUTES:
        client = empty_client if label == "/query" else _install(monkeypatch, [])
        r = _post(client, label)
        bodies[label] = (r.status_code, r.content)
    distinct = set(bodies.values())
    assert distinct == {(200, b"[]")}, (
        "the query routes disagree on the empty wire: %s" %
        {k: (s, b.decode("utf-8", "replace")) for k, (s, b) in bodies.items()}
    )


# --- The asymmetry itself: all three now carry the same declared response_model ----------------

def test_all_three_query_routes_declare_the_same_response_model():
    """The gap this card closed. If a future edit drops the model from a sibling, the asymmetry
    reopens and this reddens -- pinning the CLOSURE, not just the current behaviour."""
    from app.models import QueryHit
    from typing import List as _List

    wanted = {"/query", "/query/{entity_id}", "/query_multiple"}
    seen = {}
    for route in app.routes:
        p = getattr(route, "path", None)
        if p in wanted and "POST" in getattr(route, "methods", set()):
            seen[p] = getattr(route, "response_model", None)
    assert set(seen) == wanted, "did not find all three query POST routes: %s" % sorted(seen)
    for p, model in seen.items():
        assert model == _List[QueryHit], (
            "%s no longer declares response_model=List[QueryHit] (got %r); the sibling asymmetry "
            "#45 left has reopened." % (p, model)
        )

"""QUARANTINED on the read side: `link_state` on every `/query*` hit (CARD-P2-01 S1, part C).

THE GAP. The stored-link backfill (C:/fswt/.coord/FILES-DEV/CARD-F-EMBED-LINK-BACKFILL-2026-09-24T113102Z.md,
Demian-executed, operator-gated) QUARANTINES a pre-guard non-https citation link by MOVING it:

    SET cmetadata = (cmetadata - 'link')
        || jsonb_build_object('quarantined_link', cmetadata->>'link',
                              'quarantined_reason', 'link_scheme_not_allowed',
                              'quarantined_at', '<utc>')
    WHERE ... AND NOT (cmetadata ? 'quarantined_link')          -- its idempotency guard

After it runs, a quarantined row and a row that never had a link look the SAME to a consumer
that reads `link`: both lack it. "No source link" and "the source link was withheld for safety"
are different facts for a reader. The quarantine state existed nowhere on the read side.

THE FIX reads THAT marker -- presence of `quarantined_link`, the card's own guard predicate --
and states it as a typed, top-level `link_state` ('present' | 'quarantined' | 'none'). rag_api
writes no second marker. Top-level for the same reason as `score_kind`: `metadata` goes out
exactly as stored (the real-retrieval set-equality test depends on that), and this field
describes the stored metadata rather than being stored data.

The quarantined fixture below is the EXACT key shape the card's MUTATION statement produces --
no `link`, plus the three `quarantined_*` keys -- written out rather than imported, because the
card lives outside this repository and CI cannot read it.

RED-FIRST: on main `afd44b9` the hit has no `link_state` key, so every assertion on it fails.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

os.environ.setdefault("JWT_SECRET", "testsecret")

from app.routes import document_routes  # noqa: E402
from main import app  # noqa: E402

ROUTES = {
    "/query": ("/query", {"query": "q", "file_id": "f1", "k": 1, "entity_id": "userA"}),
    "/query/{entity_id}": ("/query/userA", {"query": "q", "k": 1}),
    "/query_multiple": ("/query_multiple", {"query": "q", "file_ids": ["f1"], "k": 1}),
}

_BASE = {"file_id": "f1", "user_id": "userA", "tenant_id": "tenantA", "filename": "brief.pdf", "page": 2}

#: What the backfill's MUTATION leaves on a row that had `link = 'http://raw-bucket.example/x'`.
QUARANTINED = {
    **_BASE,
    "quarantined_link": "http://raw-bucket.example/x",
    "quarantined_reason": "link_scheme_not_allowed",
    "quarantined_at": "2026-09-24T20:00:00Z",
}
PRESENT = {**_BASE, "link": "https://tenant.sharepoint.com/sites/x/brief.pdf"}
NONE = dict(_BASE)

CASES = {
    "quarantined": (QUARANTINED, "quarantined"),
    "present": (PRESENT, "present"),
    "none": (NONE, "none"),
    # A `link` that is empty is not a link (the write path never stores one; the dry-run SQL
    # treats '' as absent too).
    "empty_link": ({**_BASE, "link": ""}, "none"),
    # Not producible by the card's SQL (it removes `link` in the same statement). If it ever
    # happens, the row is under quarantine and must not be reported as carrying a usable link.
    "quarantined_and_link": ({**QUARANTINED, "link": "https://x.example/y"}, "quarantined"),
}


def _auth():
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"], "act": ["read"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")}


def _client(monkeypatch, metadata):
    async def fake_retrieve(*_a, **_k):
        return [(Document(page_content="the passage", metadata=dict(metadata)), 0.25)]

    monkeypatch.setattr(document_routes, "_retrieve_documents", fake_retrieve)
    monkeypatch.setattr(document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3])
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


def _hit(client, label):
    path, body = ROUTES[label]
    r = client.post(path, json=body, headers=_auth())
    assert r.status_code == 200, r.text
    hits = r.json()
    assert hits, "no hit came back, so every assertion here would pass for the wrong reason"
    return hits[0][0]


@pytest.mark.parametrize("label", list(ROUTES))
@pytest.mark.parametrize("case", list(CASES))
def test_link_state_distinguishes_quarantined_from_absent_on_every_route(monkeypatch, label, case):
    stored, expected = CASES[case]
    doc = _hit(_client(monkeypatch, stored), label)
    assert doc.get("link_state") == expected, (
        "%s / %s: link_state is %r, expected %r. Without it a quarantined link and no link at "
        "all are the same absence to a consumer." % (label, case, doc.get("link_state"), expected)
    )


@pytest.mark.parametrize("label", list(ROUTES))
def test_link_state_is_top_level_and_metadata_goes_out_exactly_as_stored(monkeypatch, label):
    """The typed state DESCRIBES the stored metadata; it is not added to it. `metadata` is the
    stored data consumers forward and persist, and the real-retrieval suite pins that nothing is
    invented inside it (test_incomplete_state_through_real_retrieval, set equality)."""
    doc = _hit(_client(monkeypatch, QUARANTINED), label)
    assert "link_state" not in doc["metadata"], doc["metadata"]
    assert doc["metadata"] == QUARANTINED, doc["metadata"]


def test_the_three_routes_agree_byte_for_byte_on_a_quarantined_hit(monkeypatch):
    """One shaping seam (`_on_the_wire`) -- so the three consumers of one producer cannot disagree
    about the quarantine state."""
    bodies = {}
    for label, (path, body) in ROUTES.items():
        bodies[label] = _client(monkeypatch, QUARANTINED).post(path, json=body, headers=_auth()).content
    assert len(set(bodies.values())) == 1, bodies
    assert b'"link_state":"quarantined"' in next(iter(bodies.values()))


def test_the_marker_read_is_the_backfill_cards_marker():
    """The read side keys on the card's marker by NAME. If the card's MUTATION ever renames the
    key, this literal is the one place that must move with it -- and this test is where the
    mismatch surfaces, instead of a quarantined row silently reading as 'none'."""
    assert document_routes.QUARANTINED_LINK_KEY == "quarantined_link"
    assert document_routes._link_state({"quarantined_link": "x"}) == "quarantined"
    assert document_routes._link_state({"quarantined_reason": "link_scheme_not_allowed"}) == "none", (
        "the marker is `quarantined_link` (the card's idempotency predicate), not the reason key"
    )

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

THE FIX reads THAT marker -- presence of `quarantined_link` -- and states it as a typed,
top-level `link_state` ('present' | 'quarantined' | 'none').

DECISION (2) (INTEGRATION, after S1 was accepted): a quarantined marker must NEVER carry the raw
refused URL, at rest OR on the wire -- it can hold credentials, a token, or a `javascript:`/`data:`
payload, and untrustworthiness is the whole reason it was quarantined. The marker is the REDACTED
object `{scheme, host, refusal_reason, sha256}` (host from urlparse.hostname, so no userinfo; sha256
of the raw for correlation, one-way). "retrieved == stored" still holds because the redaction-aware
backfill stores the OBJECT at rest and the wire serves that object unchanged; a LEGACY row that still
holds the raw string is redacted ON EMIT, so the raw never reaches the wire even before migration.

The fixtures below cover both at-rest shapes; they are written out rather than imported because the
backfill card lives outside this repository and CI cannot read it.

RED-FIRST: on `43bb826` the raw `quarantined_link` string passed through verbatim, so the
'no raw substring' / redacted-object assertions fail there; on `afd44b9` there is no `link_state`
at all.
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

import hashlib

#: A raw refused URL carrying exactly the things a quarantine marker must NEVER leak: credentials,
#: a token in the query, and a private path. Deliberately distinctive so the "no raw substring"
#: assertion is not vacuous.
RAW_URL = "http://user:s3cr3t@raw-bucket.example/secret-path/file.pdf?token=abc123XYZ"
RAW_SHA256 = hashlib.sha256(RAW_URL.encode("utf-8")).hexdigest()
#: The parts of RAW_URL that must appear NOWHERE in a served hit (host is kept, so it is not here).
RAW_LEAK_MARKERS = ("s3cr3t", "secret-path", "file.pdf", "token=abc123XYZ", RAW_URL)

#: The redacted marker RAW_URL reduces to (INTEGRATION decision (2)).
REDACTED = {
    "scheme": "http",
    "host": "raw-bucket.example",
    "refusal_reason": "link_scheme_not_allowed",
    "sha256": RAW_SHA256,
}

#: LEGACY at rest: an earlier backfill moved the raw string verbatim. Redacted ON EMIT.
QUARANTINED_RAW_AT_REST = {
    **_BASE,
    "quarantined_link": RAW_URL,
    "quarantined_reason": "link_scheme_not_allowed",
    "quarantined_at": "2026-09-24T20:00:00Z",
}
#: NEW at rest: the redaction-aware backfill stores the object. Served unchanged (retrieved==stored).
QUARANTINED_REDACTED_AT_REST = {
    **_BASE,
    "quarantined_link": dict(REDACTED),
    "quarantined_at": "2026-09-24T20:00:00Z",
}
PRESENT = {**_BASE, "link": "https://tenant.sharepoint.com/sites/x/brief.pdf"}
NONE = dict(_BASE)

CASES = {
    "quarantined_raw_at_rest": (QUARANTINED_RAW_AT_REST, "quarantined"),
    "quarantined_redacted_at_rest": (QUARANTINED_REDACTED_AT_REST, "quarantined"),
    "present": (PRESENT, "present"),
    "none": (NONE, "none"),
    # A `link` that is empty is not a link (the write path never stores one; the dry-run SQL
    # treats '' as absent too).
    "empty_link": ({**_BASE, "link": ""}, "none"),
    # Not producible by the card's SQL (it removes `link` in the same statement). If it ever
    # happens, the row is under quarantine and must not be reported as carrying a usable link.
    "quarantined_and_link": ({**QUARANTINED_RAW_AT_REST, "link": "https://x.example/y"}, "quarantined"),
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
@pytest.mark.parametrize("case", ["present", "none"])
def test_non_quarantined_metadata_goes_out_exactly_as_stored(monkeypatch, label, case):
    """The typed state DESCRIBES the stored metadata; it is not added to it. For a hit with no raw
    quarantine to redact, `metadata` is byte-for-byte what was stored -- the real-retrieval suite
    pins that nothing is invented inside it (test_incomplete_state_through_real_retrieval)."""
    stored, _ = CASES[case]
    doc = _hit(_client(monkeypatch, stored), label)
    assert "link_state" not in doc["metadata"], doc["metadata"]
    assert doc["metadata"] == stored, doc["metadata"]


# ---------------------------------------------------------------------------
# Decision (2): a quarantined marker NEVER carries the raw URL, at rest or on the wire.
# RED-FIRST on 43bb826, where the raw `quarantined_link` string passes through verbatim.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(ROUTES))
def test_a_quarantined_hit_never_carries_the_raw_url_and_carries_the_redacted_marker(monkeypatch, label):
    """RED-FIRST for decision (2). A LEGACY row holds the raw refused URL at rest; the served hit
    must carry the redacted marker only. On 43bb826 the raw string rode the wire verbatim, so the
    'no raw substring' assertions fail there; here the raw never leaves the process."""
    client = _client(monkeypatch, QUARANTINED_RAW_AT_REST)
    path, body = ROUTES[label]
    r = client.post(path, json=body, headers=_auth())
    assert r.status_code == 200, r.text

    # The whole serialized hit must not contain any raw-only fragment (host is kept, so excluded).
    raw_text = r.text
    for marker in RAW_LEAK_MARKERS:
        assert marker not in raw_text, (
            "%s: the raw quarantined URL fragment %r reached the wire: %s" % (label, marker, raw_text)
        )

    doc = r.json()[0][0]
    assert doc["link_state"] == "quarantined"
    ql = doc["metadata"]["quarantined_link"]
    assert isinstance(ql, dict), "quarantined_link must be the redacted object, not a string: %r" % ql
    assert set(ql) == {"scheme", "host", "refusal_reason", "sha256"}, ql
    assert ql == REDACTED, ql
    # sha256 correlates two rows carrying the same refused link without revealing it.
    assert ql["sha256"] == RAW_SHA256 and len(ql["sha256"]) == 64


@pytest.mark.parametrize("label", list(ROUTES))
def test_redacted_at_rest_is_served_unchanged_retrieved_equals_stored(monkeypatch, label):
    """When the redaction-aware backfill has already stored the object, the served metadata equals
    the stored metadata (retrieved == stored on the redacted form) -- the transform is a no-op."""
    doc = _hit(_client(monkeypatch, QUARANTINED_REDACTED_AT_REST), label)
    assert doc["link_state"] == "quarantined"
    assert doc["metadata"] == QUARANTINED_REDACTED_AT_REST, doc["metadata"]
    assert doc["metadata"]["quarantined_link"] == REDACTED


def test_credentials_and_a_missing_host_are_both_handled():
    """`urlparse.hostname` drops userinfo, so credentials never survive into `host`; a hostless
    scheme (javascript:, data:) redacts to host=None with the raw still gone."""
    creds = document_routes._redact_quarantined_link("http://user:pw@h.example/p?t=1")
    assert creds["host"] == "h.example" and "user" not in str(creds) and "pw" not in str(creds)
    js = document_routes._redact_quarantined_link("javascript:alert(document.cookie)")
    assert js["scheme"] == "javascript" and js["host"] is None
    assert "alert" not in str(js) and "cookie" not in str(js)
    assert len(js["sha256"]) == 64


def test_the_three_routes_agree_byte_for_byte_on_a_quarantined_hit(monkeypatch):
    """One shaping seam (`_on_the_wire`) -- so the three consumers of one producer cannot disagree
    about the quarantine state, and all three redact identically."""
    bodies = {}
    for label, (path, body) in ROUTES.items():
        bodies[label] = _client(monkeypatch, QUARANTINED_RAW_AT_REST).post(
            path, json=body, headers=_auth()
        ).content
    assert len(set(bodies.values())) == 1, bodies
    one = next(iter(bodies.values()))
    assert b'"link_state":"quarantined"' in one
    assert RAW_URL.encode() not in one and b"token=abc123XYZ" not in one


def test_the_marker_read_is_the_backfill_cards_marker():
    """The read side keys on the card's marker by NAME. If the card's MUTATION ever renames the
    key, this literal is the one place that must move with it -- and this test is where the
    mismatch surfaces, instead of a quarantined row silently reading as 'none'."""
    assert document_routes.QUARANTINED_LINK_KEY == "quarantined_link"
    assert document_routes._link_state({"quarantined_link": "x"}) == "quarantined"
    assert document_routes._link_state({"quarantined_link": dict(REDACTED)}) == "quarantined", (
        "the redacted object at rest is still the quarantine marker"
    )
    assert document_routes._link_state({"quarantined_reason": "link_scheme_not_allowed"}) == "none", (
        "the marker is `quarantined_link` (the card's idempotency predicate), not the reason key"
    )

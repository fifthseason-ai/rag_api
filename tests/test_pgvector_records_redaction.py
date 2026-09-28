"""RV-197D item (5): the DEBUG pgvector record dumps must fail-closed redact `cmetadata`.

`GET /records` and `GET /records/all` (app/routes/pgvector_routes.py) do a raw `SELECT *`
including `cmetadata` and return it. They are mounted only when DEBUG_RAG_API is truthy and are
entitlement-filtered, but at b7b2784 they are NOT redacted -- so a legacy row whose
`quarantined_link` is still the raw string leaks the raw URL/credentials/path/token here exactly
as GET /documents did before the #116 F1 fix. This applies the SAME #116 redactor
(`document_routes._redacted_metadata`, reused not re-implemented) to each served row.

The routes are exercised DIRECTLY (a fake asyncpg pool + a fake Request carrying the entitlement),
which is the seam-level test the package permits: wiring DEBUG_RAG_API at import time in the
TestClient harness is impractical (main.py mounts the router only if debug_mode was truthy when
main imported). Calling the route coroutine runs the REAL handler body, so the missing-redaction
defect reproduces at b7b2784.

RED-FIRST at b7b2784:
  * a legacy raw `quarantined_link` row served via /records and /records/all rides the wire
    verbatim -> the 'no raw substring' + 'quarantined_link is the redacted object' assertions RED.
"""
import asyncio
import hashlib
import json

import pytest

from app.routes import pgvector_routes as pv
from app.services.database import PSQLDatabase

RAW_URL = "http://user:s3cr3tPW@raw-bucket.example/secret-path/file.pdf?token=tokXYZ123"
RAW_SHA256 = hashlib.sha256(RAW_URL.encode("utf-8")).hexdigest()
#: Fragments that must appear NOWHERE in a served response (the host is kept, so it is excluded).
RAW_LEAK_MARKERS = ("s3cr3tPW", "secret-path", "file.pdf", "token=tokXYZ123", RAW_URL)
CANONICAL_KEYS = {"scheme", "host", "refusal_reason", "sha256"}

ENT = {"entity_ids": {"userA"}, "tenant_id": "tenantA", "actions": {"read"}}

#: LEGACY at rest: an earlier backfill moved the raw string verbatim into cmetadata.
LEGACY_RAW_META = {
    "user_id": "userA", "tenant_id": "tenantA", "filename": "brief.pdf",
    "quarantined_link": RAW_URL, "quarantined_reason": "link_scheme_not_allowed",
}
#: The redacted object the redaction-aware backfill writes at rest (canonical): must pass untouched.
CANONICAL_META = {
    "user_id": "userA", "tenant_id": "tenantA",
    "quarantined_link": {"scheme": "http", "host": "raw-bucket.example",
                         "refusal_reason": "link_scheme_not_allowed", "sha256": RAW_SHA256},
}


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def fetch(self, query, *args):
        # asyncpg Records are dict()-able; hand back independent dicts, as the route does.
        return [dict(r) for r in self._rows]


class _FakePool:
    def __init__(self, rows):
        self._rows = rows

    def acquire(self):
        return _FakeConn(self._rows)


class _Req:
    def __init__(self, entitlement):
        self.state = type("S", (), {})()
        self.state.entitlement = entitlement


def _install_pool(monkeypatch, rows):
    async def _fake_get_pool():
        return _FakePool(rows)

    monkeypatch.setattr(PSQLDatabase, "get_pool", _fake_get_pool)


def _call_all(rows, ent=ENT):
    return asyncio.run(pv.get_all_records(_Req(ent), table_name="langchain_pg_embedding"))


def _call_filtered(rows, ent=ENT):
    return asyncio.run(
        pv.get_records_filtered_by_custom_id(_Req(ent), custom_id="c1",
                                             table_name="langchain_pg_embedding"))


def _rows(cmeta):
    return [{"custom_id": "c1", "uuid": "u1", "cmetadata": cmeta}]


# ---------------------------------------------------------------------------
# RED-FIRST: both exits redact a legacy raw quarantined_link.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("caller", [_call_all, _call_filtered], ids=["records_all", "records"])
@pytest.mark.parametrize("as_str", [False, True], ids=["dict_cmeta", "json_str_cmeta"])
def test_records_routes_redact_a_legacy_raw_quarantined_link(monkeypatch, caller, as_str):
    """RED-FIRST at b7b2784. cmetadata arrives as a dict OR a JSON string (asyncpg JSONB); on both
    exits and both shapes the raw refused URL must NOT appear anywhere, and quarantined_link must be
    the redacted object. At b7b2784 the routes applied no redaction, so the raw rode the wire."""
    cmeta = json.dumps(LEGACY_RAW_META) if as_str else dict(LEGACY_RAW_META)
    rows = _rows(cmeta)
    _install_pool(monkeypatch, rows)

    result = caller(rows)
    blob = json.dumps(result)
    for marker in RAW_LEAK_MARKERS:
        assert marker not in blob, "records route leaked a raw fragment %r: %s" % (marker, blob)

    meta = result[0]["cmetadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    ql = meta["quarantined_link"]
    assert isinstance(ql, dict), "quarantined_link must be the redacted object, not a string: %r" % ql
    assert set(ql) == CANONICAL_KEYS, ql
    assert ql["host"] == "raw-bucket.example" and ql["scheme"] == "http", ql
    assert ql["sha256"] == RAW_SHA256, ql


@pytest.mark.parametrize("caller", [_call_all, _call_filtered], ids=["records_all", "records"])
def test_records_routes_pass_a_canonical_marker_untouched(monkeypatch, caller):
    """No-regression: an already-canonical row is served unchanged (retrieved == stored), so the
    redaction seam adds no churn to rows written by the redaction-aware backfill."""
    rows = _rows(dict(CANONICAL_META))
    _install_pool(monkeypatch, rows)
    result = caller(rows)
    assert result[0]["cmetadata"]["quarantined_link"] == CANONICAL_META["quarantined_link"]


@pytest.mark.parametrize("caller", [_call_all, _call_filtered], ids=["records_all", "records"])
def test_records_routes_drop_no_row_when_redacting(monkeypatch, caller):
    """Redaction rewrites a row in place; it must never drop or duplicate one (the entitlement
    filter is the only thing that removes rows)."""
    rows = _rows(dict(LEGACY_RAW_META))
    _install_pool(monkeypatch, rows)
    result = caller(rows)
    assert len(result) == 1, result
    assert result[0]["custom_id"] == "c1"


def test_records_all_still_filters_cross_entity_rows_then_redacts(monkeypatch):
    """The redaction runs AFTER the entitlement filter: a cross-entity row is dropped (not merely
    redacted), and the surviving own-entity legacy row is redacted."""
    rows = [
        {"custom_id": "mine", "cmetadata": dict(LEGACY_RAW_META)},
        {"custom_id": "other", "cmetadata": {"user_id": "userB", "tenant_id": "tenantA",
                                             "quarantined_link": RAW_URL}},
    ]
    _install_pool(monkeypatch, rows)
    result = _call_all(rows)
    assert {r["custom_id"] for r in result} == {"mine"}, result
    blob = json.dumps(result)
    for marker in RAW_LEAK_MARKERS:
        assert marker not in blob, blob


def test_redact_row_seam_reuses_the_document_routes_redactor():
    """Unit contract for the reused seam (present only after the fix): a legacy raw row is redacted,
    a canonical row is returned unchanged (same object -> no churn), a clean row is untouched, and a
    non-dict / unparseable cmetadata is left as-is. Skips cleanly at b7b2784 (seam absent)."""
    if not hasattr(pv, "_redact_row_cmetadata"):
        pytest.skip("seam not present at this tree (b7b2784)")
    legacy = _rows(dict(LEGACY_RAW_META))
    out = pv._redact_row_cmetadata(legacy)
    assert set(out[0]["cmetadata"]["quarantined_link"]) == CANONICAL_KEYS
    assert RAW_URL not in json.dumps(out)

    canonical = _rows(dict(CANONICAL_META))
    out2 = pv._redact_row_cmetadata(canonical)
    assert out2[0] is canonical[0], "a canonical row must pass through unchanged (no churn)"

    clean = _rows({"user_id": "userA"})
    assert pv._redact_row_cmetadata(clean)[0] is clean[0]

    weird = _rows(12345)  # not a dict, not a JSON object string
    assert pv._redact_row_cmetadata(weird)[0] is weird[0]

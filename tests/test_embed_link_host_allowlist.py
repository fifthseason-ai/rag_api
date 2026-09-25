"""A caller-supplied `link` must be on the governed-HOST allowlist, when one is configured.

CARD-P2-01 S4, G1-A -- the PRODUCER-SIDE governed-host allowlist (RV-145 N1), the owed half of
CORE-CITATION-GOVERNED-LINK. `#556` shipped the scheme/storage/presign half in Core; the
host-level rule (`RAG_GOVERNED_LINK_HOSTS`) had no source of truth in rag_api and was deferred.
This closes it on the producer side: a scheme-valid https link is trusted by HOST, not merely by
scheme, whenever an operator has configured the allowlist.

CONSERVATIVE DEFAULT, deliberately (CARD-P2-01 §11, OPEN FOR RICHARD): the allowlist VALUE is a
Richard policy/disclosure call, so when `RAG_GOVERNED_LINK_HOSTS` is UNSET/empty the host check is
DISABLED and the pre-existing https-only behaviour is preserved with ZERO new rejections. The
tests below prove BOTH halves: the SET behaviour and the UNSET (no-new-rejections) default.

Matching contract (documented at `_governed_link_hosts` / `_host_is_governed`):
  * exact host       -- "contoso.sharepoint.com" matches only itself
  * dot-prefix suffix -- ".corp.example.com" matches any subdomain, NOT the bare parent, and NOT a
    look-alike ("evilcorp.example.com") -- because that string does not end with the leading dot
  * hostname only    -- urlparse.hostname drops userinfo (credentials) and port, so a spoofed
    userinfo ("https://allowed.host@evil.com/") is matched on the REAL host, evil.com

RED-FIRST (proven on the anchor 5dfe7ad, before this branch):
  * behavioural cases patch the resolved constant with raising=False, so on 5dfe7ad the guard code
    does not exist and a non-allowlisted host ingests (200) -> the 422 assertions RED.
  * the UNSET regression is green on both (no host guard existed), pinning that the default adds
    nothing.
  * `test_the_host_allowlist_symbols_exist` reds on 5dfe7ad (symbols absent) and is the rename-loud
    pin on this branch.

Controls (mutation):
  * delete `if not allow: return` (always enforce) -> the UNSET regression reds (the default is
    what protects existing uploads).
  * neuter `_host_is_governed` to `return True` -> the non-allowlisted-host refusal reds.
  * neuter `_host_is_governed` to `return False` -> every ACCEPT-on-allowed-host case reds.
"""
import datetime
import io
import os

import jwt
import pytest
from fastapi.testclient import TestClient

import app.routes.document_routes as dr
from main import app

SECRET = "test-secret-embed-host"

client = TestClient(app)


@pytest.fixture(autouse=True)
def _store_double(monkeypatch):
    """The minimum that lets /embed reach 200 without a database (mirrors
    tests/test_embed_link_validation.py): the ACCEPT cases are only meaningful if the route can
    actually complete."""
    from concurrent.futures import ThreadPoolExecutor

    from app.services.vector_store.async_pg_vector import AsyncPgVector

    os.environ["JWT_SECRET"] = SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    async def _aadd(self, docs, ids=None, executor=None):
        return ids

    async def _filtered(self, ids, user_id=None, document_origin_type=None,
                        subscription_id=None, executor=None):
        return []

    async def _delete(self, ids=None, collection_only=False, user_id=None,
                      document_origin_type=None, subscription_id=None, executor=None, **_):
        return None

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", _aadd)
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", _filtered)
    monkeypatch.setattr(AsyncPgVector, "delete", _delete)
    yield


def _set_allowlist(monkeypatch, hosts):
    """Configure the RESOLVED allowlist the route consults. raising=False on purpose: on the
    anchor 5dfe7ad this symbol does not exist, so the behavioural cases must still RUN (and red
    on the missing guard) rather than error at setup. The symbol itself is pinned separately by
    `test_the_host_allowlist_symbols_exist`."""
    monkeypatch.setattr(dr, "_GOVERNED_LINK_HOSTS", tuple(hosts), raising=False)


def hdr(uid="userA", tid="tenantA"):
    os.environ["JWT_SECRET"] = SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ["userA"],
        "act": ["write"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, SECRET, algorithm='HS256')}"}


def _file():
    return {"file": ("t.txt", io.BytesIO(b"hello world"), "text/plain")}


def _embed(link=None, file_id="f-host"):
    data = {"file_id": file_id, "entity_id": "userA"}
    if link is not None:
        data["link"] = link
    return client.post("/embed", data=data, files=_file(), headers=hdr())


# --- allowlist SET -------------------------------------------------------------------------

ALLOW = ("contoso.sharepoint.com", ".corp.example.com")


def test_a_non_allowlisted_https_host_is_refused_with_a_typed_reason(monkeypatch):
    """THE REFUSAL. With an allowlist configured, an https link on a host that is not on it is
    refused with a 422 and the NEW machine-readable reason -- the same envelope shape as
    link_scheme_not_allowed."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://arbitrary-lookalike.example.org/x/Deck.pptx")

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["link"]["reason"] == "link_host_not_allowed", detail
    assert "message" in detail and detail["message"], detail
    assert detail["link"]["allowed_hosts"] == list(ALLOW), detail


def test_an_allowlisted_exact_host_passes(monkeypatch):
    """The exact host Core's connector sync sends must ingest untouched."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://contoso.sharepoint.com/sites/x/Shared%20Documents/Deck.pptx")
    assert r.status_code == 200, r.text


def test_a_dot_prefixed_subdomain_of_an_allowlisted_domain_passes(monkeypatch):
    """A dot-prefix entry admits its subdomains."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://team-a.corp.example.com/docs/Report.pdf")
    assert r.status_code == 200, r.text


def test_the_bare_parent_of_a_dot_prefix_entry_is_not_admitted(monkeypatch):
    """`.corp.example.com` admits subdomains but NOT the bare parent -- the parent would need its
    own exact entry. Proves the dot-prefix is a subdomain rule, not a substring rule."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://corp.example.com/docs/Report.pdf")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_host_not_allowed", r.text


def test_a_suffix_lookalike_of_a_dot_prefix_entry_is_refused(monkeypatch):
    """`.corp.example.com` must not admit `evilcorp.example.com` -- that string ends with
    `corp.example.com` but NOT with the leading dot, so it is a different host."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://evilcorp.example.com/x")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_host_not_allowed", r.text


def test_userinfo_and_port_are_ignored_the_match_is_on_the_hostname(monkeypatch):
    """A link with credentials in the userinfo and a non-default port is matched on the HOSTNAME
    alone (urlparse.hostname drops both), so an allowlisted host still passes."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://user:secret@contoso.sharepoint.com:8443/sites/x/Deck.pptx")
    assert r.status_code == 200, r.text


def test_userinfo_spoofing_the_allowed_host_does_not_bypass_the_real_host(monkeypatch):
    """`https://contoso.sharepoint.com@evil.com/` resolves to the REAL host evil.com. Matching on
    userinfo instead of hostname would be a bypass; the real host is refused."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://contoso.sharepoint.com@evil.com/x")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["link"]["reason"] == "link_host_not_allowed", detail
    assert detail["link"]["host"] == "evil.com", detail


def test_the_match_is_case_insensitive(monkeypatch):
    """Hosts are case-folded (urlparse lower-cases the hostname), so an upper-cased host on the
    allowlist still passes -- DNS is case-insensitive."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://CONTOSO.SharePoint.com/sites/x/Deck.pptx")
    assert r.status_code == 200, r.text


def test_the_raw_refused_link_is_never_echoed_back(monkeypatch):
    """The refusal body reports only the redacted host, reason and allowed list -- never the raw
    link, which can carry a token in the query or credentials in the userinfo."""
    _set_allowlist(monkeypatch, ALLOW)
    payload = "https://sneaky-canary-host@evil.com/p/sneaky-canary-path?token=sneaky-canary-tok#sneaky-canary-frag"
    r = _embed(link=payload)

    assert r.status_code == 422, r.text
    assert "sneaky-canary-tok" not in r.text, r.text
    assert "sneaky-canary-path" not in r.text, r.text
    assert "sneaky-canary-host" not in r.text, r.text
    assert "sneaky-canary-frag" not in r.text, r.text
    # the SAFE redacted host is what is surfaced
    assert r.json()["detail"]["link"]["host"] == "evil.com", r.text


def test_the_refused_host_is_never_echoed_into_the_log(monkeypatch, caplog):
    """The never-echo rule holds in the LOGS too. The log records the bounded hostname only, never
    the raw link (query/userinfo/path)."""
    import logging

    _set_allowlist(monkeypatch, ALLOW)
    payload = "https://logcanary-user@evil.com/logcanary-path?token=logcanary-tok"
    with caplog.at_level(logging.DEBUG):
        r = _embed(link=payload)

    assert r.status_code == 422, r.text
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "logcanary-tok" not in joined, joined
    assert "logcanary-path" not in joined, joined
    assert "logcanary-user" not in joined, joined
    refusals = [rec for rec in caplog.records
                if "refused a link whose host is not allowed" in rec.getMessage()]
    assert len(refusals) == 1, [rec.getMessage() for rec in caplog.records]
    assert refusals[0].levelno == logging.WARNING, refusals[0].levelname


def test_a_refused_host_reaches_no_upload_extraction_or_store(monkeypatch):
    """PLACEMENT: the host refusal is at the route boundary, so a bad host costs no work -- no
    temp file, no loader run. Moving the check after extraction would red this."""
    _set_allowlist(monkeypatch, ALLOW)
    calls = []
    monkeypatch.setattr(dr, "_make_unique_temp_path",
                        lambda *a, **k: calls.append("temp") or "/tmp/should-not-be-used")
    monkeypatch.setattr(dr, "get_loader",
                        lambda *a, **k: calls.append("loader") or (_ for _ in ()).throw(AssertionError("loader ran")))

    r = _embed(link="https://arbitrary.example.org/x")
    assert r.status_code == 422, r.text
    assert calls == [], "a refused host did work before refusing: %r" % calls


def test_the_reported_host_is_length_capped(monkeypatch):
    """The hostname is caller-influenced text reflected back; cap its length (a real FQDN is
    <= 253 chars)."""
    _set_allowlist(monkeypatch, ALLOW)
    long_label = "a" * 300
    r = _embed(link=f"https://{long_label}.example.org/x")
    assert r.status_code == 422, r.text
    reported = r.json()["detail"]["link"]["host"]
    assert len(reported) <= 253, len(reported)


# --- scheme guard still comes FIRST, even with a host allowlist configured -----------------

@pytest.mark.parametrize("bad_link, expected_scheme", [
    ("javascript:alert(1)", "javascript"),
    ("data:text/html;base64,PHNjcmlwdD4=", "data"),
    ("http://contoso.sharepoint.com/x/Deck.pptx", "http"),
])
def test_scheme_guard_runs_before_the_host_guard(monkeypatch, bad_link, expected_scheme):
    """A non-https link is refused as link_scheme_not_allowed regardless of the host rule -- even
    when the host would be on the allowlist (the http case). The scheme is the first gate."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link=bad_link)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_scheme_not_allowed", r.text


# --- allowlist UNSET: the conservative default adds ZERO new rejections --------------------

def test_when_the_allowlist_is_unset_an_arbitrary_https_host_still_ingests(monkeypatch):
    """THE CONSERVATIVE DEFAULT (regression guard). With the allowlist empty -- the shipped
    default, since the value is OPEN FOR RICHARD -- the host check is disabled and any https host
    ingests exactly as before. This is what must stay true so S4 changes nothing until an operator
    opts in."""
    _set_allowlist(monkeypatch, ())
    r = _embed(link="https://any-arbitrary-host.example.net/whatever/Deck.pptx")
    assert r.status_code == 200, r.text


def test_when_the_allowlist_is_unset_a_non_https_link_is_still_refused_on_scheme(monkeypatch):
    """The default disables only the HOST check; the scheme guard is unconditional."""
    _set_allowlist(monkeypatch, ())
    r = _embed(link="http://any-arbitrary-host.example.net/x")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_scheme_not_allowed", r.text


def test_an_upload_without_a_link_is_unaffected_by_a_configured_allowlist(monkeypatch):
    """ABSENT IS NOT A REFUSAL, even with a host allowlist set: organic uploads send no link."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link=None)
    assert r.status_code == 200, r.text


# --- unit-level matching + parsing semantics ----------------------------------------------

def test_the_host_allowlist_symbols_exist():
    """RENAME-LOUD PIN (reds on 5dfe7ad, where these do not exist). Names the contract the guard
    relies on: the resolved constant and the matcher."""
    assert hasattr(dr, "_GOVERNED_LINK_HOSTS")
    assert hasattr(dr, "_host_is_governed")
    assert hasattr(dr, "_governed_link_hosts")


@pytest.mark.parametrize("host, allow, expected", [
    ("contoso.sharepoint.com", ("contoso.sharepoint.com",), True),   # exact
    ("other.sharepoint.com", ("contoso.sharepoint.com",), False),    # exact: no substring
    ("a.corp.example.com", (".corp.example.com",), True),            # dot-prefix subdomain
    ("deep.a.corp.example.com", (".corp.example.com",), True),       # deeper subdomain
    ("corp.example.com", (".corp.example.com",), False),             # bare parent not admitted
    ("evilcorp.example.com", (".corp.example.com",), False),         # suffix look-alike refused
    ("CONTOSO.SHAREPOINT.COM", ("contoso.sharepoint.com",), True),   # case-folded
    ("", ("contoso.sharepoint.com",), False),                        # no host
    ("contoso.sharepoint.com", (), False),                           # empty allowlist admits nothing
])
def test_host_is_governed_matching_semantics(host, allow, expected):
    assert dr._host_is_governed(host, allow) is expected


@pytest.mark.parametrize("raw, expected", [
    ("", ()),
    ("   ", ()),
    ("contoso.sharepoint.com", ("contoso.sharepoint.com",)),
    ("a.com,b.com", ("a.com", "b.com")),
    ("a.com b.com", ("a.com", "b.com")),
    ("a.com, b.com\t.c.com", ("a.com", "b.com", ".c.com")),
    ("A.COM", ("a.com",)),                                            # lower-cased
    (" a.com ,, b.com ", ("a.com", "b.com")),                        # empties dropped
])
def test_governed_link_hosts_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("RAG_GOVERNED_LINK_HOSTS", raw)
    assert dr._governed_link_hosts() == expected


def test_governed_link_hosts_defaults_to_empty_when_env_absent(monkeypatch):
    """UNSET env => empty tuple => host check disabled. The conservative default has no source but
    the operator's own configuration; nothing is invented."""
    monkeypatch.delenv("RAG_GOVERNED_LINK_HOSTS", raising=False)
    assert dr._governed_link_hosts() == ()

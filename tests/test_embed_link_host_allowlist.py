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


def test_a_non_default_port_is_ignored_the_match_is_on_the_hostname(monkeypatch):
    """A non-default port is matched on the HOSTNAME alone (urlparse.hostname drops the port), so
    an allowlisted host still passes. (RV-197 F2 CHANGE: userinfo, which this test used to also
    cover, is now REFUSED outright rather than ignored -- see the malformed-authority tests below.
    The port half of the original assertion is preserved here.)"""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://contoso.sharepoint.com:8443/sites/x/Deck.pptx")
    assert r.status_code == 200, r.text


def test_userinfo_spoofing_the_allowed_host_does_not_bypass_the_real_host(monkeypatch):
    """`https://contoso.sharepoint.com@evil.com/` puts the allowed-looking string in the USERINFO;
    the real host is evil.com. RV-197 F2: any userinfo is now refused OUTRIGHT before the host
    check, so this can never bypass -- and it is refused with the malformed-authority reason, not
    resolved by a host lookup that a parser differential could get wrong. (Previously this returned
    422 link_host_not_allowed via urlparse.hostname; the refusal is now earlier and unconditional
    on userinfo.)"""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://contoso.sharepoint.com@evil.com/x")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_malformed_authority", r.text


def test_the_match_is_case_insensitive(monkeypatch):
    """Hosts are case-folded (urlparse lower-cases the hostname), so an upper-cased host on the
    allowlist still passes -- DNS is case-insensitive."""
    _set_allowlist(monkeypatch, ALLOW)
    r = _embed(link="https://CONTOSO.SharePoint.com/sites/x/Deck.pptx")
    assert r.status_code == 200, r.text


def test_the_raw_refused_link_is_never_echoed_back(monkeypatch):
    """The HOST refusal body reports only the redacted host, reason and allowed list -- never the
    raw link, which can carry a token in the query. (RV-197 F2 CHANGE: this used to use a userinfo
    payload to reach the host branch; userinfo is now refused earlier as malformed authority -- its
    own never-echo test is test_the_malformed_authority_refusal_never_echoes_the_raw_link -- so this
    now uses a plain non-allowlisted host, which is the branch it was always exercising.)"""
    _set_allowlist(monkeypatch, ALLOW)
    payload = "https://sneaky-canary-host.example.org/p/sneaky-canary-path?token=sneaky-canary-tok#sneaky-canary-frag"
    r = _embed(link=payload)

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_host_not_allowed", r.text
    assert "sneaky-canary-tok" not in r.text, r.text
    assert "sneaky-canary-path" not in r.text, r.text
    assert "sneaky-canary-frag" not in r.text, r.text
    # the SAFE redacted host is what is surfaced (the host itself is not a secret)
    assert r.json()["detail"]["link"]["host"] == "sneaky-canary-host.example.org", r.text


def test_the_refused_host_is_never_echoed_into_the_log(monkeypatch, caplog):
    """The never-echo rule holds in the LOGS too on the HOST refusal. The log records the bounded
    hostname only, never the raw link (query/path). (RV-197 F2 CHANGE: uses a plain non-allowlisted
    host now that userinfo is intercepted earlier as malformed authority.)"""
    import logging

    _set_allowlist(monkeypatch, ALLOW)
    payload = "https://logcanary-host.example.org/logcanary-path?token=logcanary-tok"
    with caplog.at_level(logging.DEBUG):
        r = _embed(link=payload)

    assert r.status_code == 422, r.text
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "logcanary-tok" not in joined, joined
    assert "logcanary-path" not in joined, joined
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


# ===========================================================================
# RV-197 F2 SECURITY: authority parser-differential + userinfo guard.
#
# Python's urlparse and WHATWG (browsers, Node `new URL`, which Core's isGovernedUrl uses)
# disagree on `\`: Python does not treat it as a path separator, so
#   https://evil.com\@contoso.sharepoint.com/x  -> Python host contoso.sharepoint.com (ADMITTED)
#                                                  WHATWG host evil.com
#   https://evil.com\.contoso.sharepoint.com/x  -> Python host evil.com\.contoso.sharepoint.com
#                                                  (ends with .sharepoint.com -> ADMITTED)
#                                                  WHATWG host evil.com
# A stored citation would then open on evil.com while the producer counted it as governed.
# Userinfo (`https://user:secret@allowed.host/...`) is ingested and STORED with the secret.
#
# FIX: when an allowlist is configured, refuse any link whose AUTHORITY contains a backslash,
# ASCII control char, whitespace, or %-encoded delimiter, and refuse userinfo (user[:pass]@)
# outright -- BEFORE the host check, giving parser parity with Core's isGovernedUrl / ADV-2 rule.
# The scheme guard still comes first; UNSET stays scheme-only (no new rejections).
#
# RED-FIRST at aa66a3ab9: every case below returns 200 (ADMITTED) on the merged tree.
# ===========================================================================

#: Allowlist for the F2 cases (per the RV-197 spec): an exact host + a dot-prefix suffix.
F2_ALLOW = (".sharepoint.com", "contoso.sharepoint.com")

MALFORMED = "link_malformed_authority"


@pytest.mark.parametrize("bad_link", [
    "https://evil.com\\@contoso.sharepoint.com/x",   # Python host contoso.sharepoint.com; WHATWG evil.com
    "https://evil.com\\.contoso.sharepoint.com/x",   # Python host ends with .sharepoint.com; WHATWG evil.com
])
def test_a_backslash_in_the_authority_is_refused_not_admitted(monkeypatch, bad_link):
    """THE PARSER DIFFERENTIAL. A backslash in the authority is a WHATWG path separator, so
    a browser and Core resolve a DIFFERENT host than Python's urlparse. Both shapes are ADMITTED
    (200) on the merged tree; the fix refuses them as malformed authority."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link=bad_link)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


def test_a_userinfo_bearing_link_on_an_allowed_host_is_refused_and_not_stored(monkeypatch):
    """Userinfo carries a secret that would be STORED verbatim (RV-183). Even on an allowlisted
    host, a link with userinfo is refused OUTRIGHT -- matching Core's ADV-2 rule -- and reaches
    no temp file, no loader and no store write. RED-FIRST: 200 (stored) on the merged tree."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    calls = []
    monkeypatch.setattr(dr, "_make_unique_temp_path",
                        lambda *a, **k: calls.append("temp") or "/tmp/should-not-be-used")
    monkeypatch.setattr(dr, "get_loader",
                        lambda *a, **k: calls.append("loader") or (_ for _ in ()).throw(AssertionError("loader ran")))

    r = _embed(link="https://user:secret@contoso.sharepoint.com:8443/sites/x/Deck.pptx")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text
    assert calls == [], "a userinfo link did work before refusing (would have been stored): %r" % calls


def test_a_percent_encoded_delimiter_in_the_authority_is_refused(monkeypatch):
    """A %-encoded delimiter (here %5C = backslash) in the authority is the same differential wearing
    an encoding. RED-FIRST: Python host is contoso.sharepoint.com -> 200 on the merged tree."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link="https://evil.com%5C@contoso.sharepoint.com/x")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


def test_whitespace_in_the_authority_is_refused(monkeypatch):
    """A tab in the authority is stripped by Python's urlsplit (a sanitization differential), so the
    merged tree parses a clean allowed host and ADMITS it (200). The fix inspects the RAW authority
    and refuses embedded whitespace/control before urlparse can sanitize it away."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link="https://evil.com\t@contoso.sharepoint.com/x")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


@pytest.mark.parametrize("good_link", [
    "https://contoso.sharepoint.com/sites/x/Shared%20Documents/Deck.pptx",  # exact host; %20 is in the PATH
    "https://a.contoso.sharepoint.com/docs/Report.pdf",                     # dot-prefix subdomain
    "https://contoso.sharepoint.com:8443/sites/x/Deck.pptx",                # a bare port is fine
])
def test_a_legitimate_allowed_link_still_passes_after_the_authority_guard(monkeypatch, good_link):
    """NON-VACUITY / no-regression: the authority guard must not refuse a clean allowed link. A
    %-encoded space in the PATH (not the authority) and a bare port must still ingest."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link=good_link)
    assert r.status_code == 200, r.text


def test_the_malformed_authority_refusal_never_echoes_the_raw_link(monkeypatch):
    """The refusal reports the typed reason only, never the raw link (which can carry a token or
    credentials). The canary fragments must appear nowhere in the body."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    payload = "https://mauth-canary-user:mauth-canary-tok@evil.com\\@contoso.sharepoint.com/mauth-canary-path"
    r = _embed(link=payload)
    assert r.status_code == 422, r.text
    for canary in ("mauth-canary-tok", "mauth-canary-path", "mauth-canary-user"):
        assert canary not in r.text, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


def test_the_scheme_guard_still_precedes_the_authority_guard(monkeypatch):
    """Ordering: a non-https link with a backslash authority is refused on SCHEME first, so the
    scheme case is never masked by the new authority guard."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link="http://evil.com\\@contoso.sharepoint.com/x")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == "link_scheme_not_allowed", r.text


@pytest.mark.parametrize("link", [
    "https://evil.com\\@contoso.sharepoint.com/x",
    "https://user:secret@any-arbitrary-host.example.net/x",
    "https://evil.com\\.contoso.sharepoint.com/x",
])
def test_when_the_allowlist_is_unset_the_authority_guard_adds_no_new_rejections(monkeypatch, link):
    """THE CONSERVATIVE DEFAULT (regression guard). With the allowlist UNSET the whole host-governance
    layer -- including the authority guard -- is disabled, so a scheme-valid link ingests exactly as
    before. This is what must stay true so RV-197 F2 changes nothing until an operator opts in.
    (Green on both the merged tree and the fix.)"""
    _set_allowlist(monkeypatch, ())
    r = _embed(link=link)
    assert r.status_code == 200, r.text


def test_the_authority_guard_symbol_exists():
    """RENAME-LOUD PIN. Names the helper the guard relies on so a rename surfaces here."""
    assert hasattr(dr, "_reject_malformed_authority")


# ===========================================================================
# RV-197 RE-REVIEW residual + notes (SECURITY): control-char separator bypass, value-shape leak's
# sibling on the authority side, non-ASCII/punycode, and typed 422 for unparseable authorities.
# ===========================================================================

EXACT_ALLOW = ("contoso.sharepoint.com",)


def _no_work_spy(monkeypatch):
    """Spy that proves a refused link reaches no temp file, no loader and hence no store write --
    the 'nothing stored on refusal' contract. Returns the calls list (must stay empty)."""
    calls = []
    monkeypatch.setattr(dr, "_make_unique_temp_path",
                        lambda *a, **k: calls.append("temp") or "/tmp/should-not-be-used")
    monkeypatch.setattr(dr, "get_loader",
                        lambda *a, **k: calls.append("loader") or (_ for _ in ()).throw(AssertionError("loader ran")))
    return calls


# --- BLOCKING: control char inside the "://" separator (T16-T19, T36) ----------------------

#: A tab/LF/CR placed inside `://` makes the OLD `_raw_authority` (link.find("://")) return '' while
#: Python's urlsplit strips the control char and reads the ALLOWED host -> admitted 200 and STORED
#: raw; WHATWG (browser/Core new URL) reads evil.com. RED-FIRST at 1c8e690 (200 stored) under SP.
RESIDUAL_SEPARATOR_CASES = {
    "T16_tab_in_sep": "https:/\t/evil.com\\@contoso.sharepoint.com/x",
    "T17_tab_before_slashes": "https:\t//evil.com\\@contoso.sharepoint.com/x",
    "T18_lf_in_sep": "https:/\n/evil.com\\@contoso.sharepoint.com/x",
    "T19_cr_in_sep_dot": "https:/\r/evil.com\\.contoso.sharepoint.com/x",
    "T36_tab_in_sep_plus_later_scheme": "https:/\t/evil.com\\@contoso.sharepoint.com/x?next=https://ok.example/",
}


@pytest.mark.parametrize("allow", [F2_ALLOW, EXACT_ALLOW], ids=["SP_dotprefix", "EXACT"])
@pytest.mark.parametrize("case", list(RESIDUAL_SEPARATOR_CASES), ids=list(RESIDUAL_SEPARATOR_CASES))
def test_a_control_char_in_the_separator_is_refused_and_not_stored(monkeypatch, allow, case):
    """THE RV-197 RESIDUAL. Under both allowlist shapes the separator-trick links must be REFUSED
    (422 malformed) and nothing stored. RED-FIRST at 1c8e690: SP admits all five (200 stored);
    EXACT admits T16/T17/T18/T36 (200 stored) and refuses T19 on host (still red because the reason
    was link_host_not_allowed, not malformed)."""
    _set_allowlist(monkeypatch, allow)
    calls = _no_work_spy(monkeypatch)
    r = _embed(link=RESIDUAL_SEPARATOR_CASES[case])
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text
    assert calls == [], "a bypass link did work before refusing (would have been stored): %r" % calls


def test_a_control_char_anywhere_in_the_link_is_refused(monkeypatch):
    """Isolates defence (a): a control char in the PATH (authority == netloc, so defence (b) passes)
    must still be refused. RED-FIRST at 1c8e690 (tab stripped -> allowed host -> 200)."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link="https://contoso.sharepoint.com/pa\tth")
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


def test_raw_authority_must_equal_parsed_netloc(monkeypatch):
    """Isolates defence (b) at unit level: when `_raw_authority(link)` differs from `parsed.netloc`
    (and no control char is present) the guard refuses `authority_mismatch`, so the per-char scan and
    the downstream host match can never read different strings."""
    import types

    link = "https://different.host/x"                       # _raw_authority -> 'different.host'
    parsed = types.SimpleNamespace(netloc="contoso.sharepoint.com", hostname="contoso.sharepoint.com")
    with pytest.raises(Exception) as ei:
        dr._reject_malformed_authority(link, parsed, "f-b")
    detail = getattr(ei.value, "detail", {})
    assert detail.get("link", {}).get("reason") == MALFORMED, detail


# --- N8: %-only and whitespace-only cases with NO "@" (isolate MF2d / MF2e) ----------------

@pytest.mark.parametrize("bad_link,kind_hint", [
    ("https://evil.com%2e.contoso.sharepoint.com/x", "percent"),   # no '@' -> isolates the % branch
    ("https://evil.com .contoso.sharepoint.com/x", "whitespace"),  # no '@' -> isolates the whitespace branch
])
def test_percent_and_whitespace_branches_are_independently_refused(monkeypatch, bad_link, kind_hint):
    """N8: the pre-existing %/whitespace cases also contained '@', so the userinfo branch refused
    them and the %/whitespace branches had no isolating test (MF2d/MF2e survived). These @-free
    cases refuse via their own branch."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link=bad_link)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


# --- N4: non-ASCII / punycode authority (T05 invalid punycode, T20 fullwidth, T06 Cyrillic) ----

@pytest.mark.parametrize("bad_link,label", [
    ("https://xn--invalidpunycodexyz.sharepoint.com/x", "T05_punycode"),
    ("https://evil.com＼.contoso.sharepoint.com/x", "T20_fullwidth_backslash"),
    ("https://cоntoso.sharepoint.com/x", "T06_cyrillic"),
])
def test_a_non_ascii_or_punycode_authority_is_refused(monkeypatch, bad_link, label):
    """N4. The governed-host list is ASCII, and stdlib IDNA admits invalid punycode WHATWG rejects.
    CHOSEN POLICY (conservative tightening the RV-197 receipt permits for T06/T21/T29): refuse any
    non-ASCII authority and any `xn--` punycode label. T05 and T20 MUST become refused; T06 is
    refused by the same consistent policy. RED-FIRST at 1c8e690 (all 200 under SP)."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link=bad_link)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text


# --- N5: an unparseable authority is a typed 422, not an untyped 500 -----------------------

@pytest.mark.parametrize("bad_link,label", [
    ("https://contoso.sharepoint.com＠evil.com/x", "T10_fullwidth_at"),
    ("https://[contoso.sharepoint.com]/x", "T14_bad_ipv6"),
    ("https://evil.com／@contoso.sharepoint.com/x", "T22_fullwidth_slash"),
])
def test_an_unparseable_authority_is_a_typed_422_not_a_500(monkeypatch, bad_link, label):
    """N5. urlparse raises ValueError on an NFKC-unsafe / bad-IPv6 authority; at 1c8e690 the route
    answered an untyped 500. It must be a typed 422 link_malformed_authority (Rule 28), nothing
    stored, no echo. RED-FIRST at 1c8e690 (500)."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    calls = _no_work_spy(monkeypatch)
    r = _embed(link=bad_link)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["link"]["reason"] == MALFORMED, r.text
    assert calls == [], calls


# --- N6: restore the malformed-authority LOG never-echo assertion --------------------------

def test_the_malformed_authority_refusal_never_echoes_the_raw_link_into_the_log(monkeypatch, caplog):
    """N6: the dropped `logcanary-user` caplog check had no repo replacement, so no permanent test
    covered the malformed-authority LOG path. Restore it: the refusal logs exactly one warning that
    names the bounded `kind` and NONE of the raw link (userinfo/path/token)."""
    import logging

    _set_allowlist(monkeypatch, F2_ALLOW)
    payload = "https://mlog-canary-user:mlog-canary-tok@evil.com\\@contoso.sharepoint.com/mlog-canary-path"
    with caplog.at_level(logging.DEBUG):
        r = _embed(link=payload)

    assert r.status_code == 422, r.text
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    for canary in ("mlog-canary-user", "mlog-canary-tok", "mlog-canary-path"):
        assert canary not in joined, joined
    refusals = [rec for rec in caplog.records
                if "refused a link whose authority is malformed" in rec.getMessage()]
    assert len(refusals) == 1, [rec.getMessage() for rec in caplog.records]
    assert refusals[0].levelno == logging.WARNING, refusals[0].levelname


# --- no-regression: legitimate allowed links still pass after all the new checks -----------

@pytest.mark.parametrize("good_link", [
    "https://contoso.sharepoint.com/sites/x/Shared%20Documents/Deck.pptx",
    "https://a.contoso.sharepoint.com/docs/Report.pdf",
    "https://contoso.sharepoint.com:8443/sites/x/Deck.pptx",
])
def test_legitimate_links_still_pass_after_the_hardening(monkeypatch, good_link):
    """The added control-char / authority-mismatch / non-ASCII / punycode checks must not refuse a
    clean ASCII allowed link (a %-encoded space in the PATH, a subdomain, or a bare port)."""
    _set_allowlist(monkeypatch, F2_ALLOW)
    r = _embed(link=good_link)
    assert r.status_code == 200, r.text

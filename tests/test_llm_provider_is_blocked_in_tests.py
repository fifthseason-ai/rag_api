"""The test suite cannot reach the real LLM provider -- summarizer path, direct invoke, every live
holder -- and the import-time boto3 session makes no IMDS probe.

N5 (2026-09-24), the LLM twin of tests/test_embeddings_provider_is_blocked_in_tests.py and
tests/test_rerank_provider_is_blocked_in_tests.py. MEASURED (RV-150 N1): app.config defaults
LLM_PROVIDER=bedrock, so app.config.llm is a LIVE ChatBedrockConverse whose `.client` is a boto3
bedrock-runtime client. Paid inference is reachable two ways -- the summarizer (SUM_UP_KNOWLEDGE_FILES,
OFF by DEFAULT) and a direct `llm.invoke`. A default is not a guard (RV-128/N1: "unable to reach", not
"unreachable by default").

The guard is the autouse fixture `_block_the_real_llm_provider` in tests/conftest.py, which replaces
the boto3 client's socket-opening step on EVERY live holder (app.config.llm AND document_routes.llm).
This file is its CONTROL, run with a SOCKET SPY: `socket.getaddrinfo` and `socket.socket.connect` are
patched to record and refuse, so "no attempt" is asserted at the OS boundary, not inferred -- the
in-suite, permanent stand-in for `--network none` (the reviewer reproduces under a real --network none
too; the spy records at getaddrinfo, which fires even when DNS is dead).

IMDS: conftest sets AWS_EC2_METADATA_DISABLED=true and explicit dummy creds+region BEFORE app.config
imports, so botocore never consults the metadata service. `test_the_config_import_makes_no_imds_probe`
proves it (0 attempts to 169.254.169.254 while app.config is (re)imported under the spy).

Controls:
  * remove the guard's setattr in conftest        -> tests 1, 2 and 6 red (spy records an attempt)
  * test 3 is the PERMANENT deliberate-regression: a real client built via the live init_llm path
    DOES attempt the network -- so the guard is the only thing between the suite and a socket
  * remove AWS_EC2_METADATA_DISABLED / the dummy creds in conftest -> the IMDS probe test can red

N2 (RV-150): under bedrock the deliberate-regression's socket is the provider host
(bedrock-runtime.<region>.amazonaws.com); the test asserts "an attempt", which is provider-agnostic.
"""
import socket
from contextlib import contextmanager
from importlib import reload

import pytest
from langchain_core.documents import Document

import app.config as cfg
from app.routes import document_routes
from app.services.summarization import summarize_file_chunks
from tests.conftest import _BlockedBedrockRuntimeClient, _guard_every_live_llm_client, _live_bedrock_llms

_IMDS_IP = "169.254.169.254"


@contextmanager
def _socket_spy():
    """Record every attempt to resolve or connect, and refuse it. Recording at getaddrinfo means an
    attempt is captured even under a real --network none, where DNS fails before connect()."""
    attempts = []

    def _spy_getaddrinfo(host, port, *args, **kwargs):
        attempts.append(("getaddrinfo", host, port))
        raise OSError("socket spy: the network is closed in tests")

    def _spy_connect(self, address):
        attempts.append(("connect", address))
        raise OSError("socket spy: the network is closed in tests")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(socket, "getaddrinfo", _spy_getaddrinfo)
        m.setattr(socket.socket, "connect", _spy_connect)
        yield attempts


def _require_live_bedrock_llm():
    live = _live_bedrock_llms()
    assert live, (
        "no live bedrock LLM holder found -- the guard has nothing to stand on. "
        "(LLM_PROVIDER must resolve to bedrock in the test env for this control to be meaningful.)"
    )
    return live


def test_the_summarizer_path_is_blocked_before_any_socket():
    """PATH 1 -- the summarizer: summarize_file_chunks builds `PROMPT | llm | StrOutputParser()` and
    invokes it, exactly as the SUM_UP_KNOWLEDGE_FILES background task does. With the guard on it must
    raise before opening a socket."""
    _require_live_bedrock_llm()
    with _socket_spy() as attempts:
        with pytest.raises(Exception):
            summarize_file_chunks(cfg.llm, [Document(page_content="a paragraph to summarize")])
    assert attempts == [], "the summarizer path reached the network: %r" % attempts


def test_a_direct_llm_invoke_is_blocked_before_any_socket():
    """PATH 2 -- a direct llm.invoke. Same guard, same outcome: no socket."""
    _require_live_bedrock_llm()
    with _socket_spy() as attempts:
        with pytest.raises(Exception):
            cfg.llm.invoke("hello")
    assert attempts == [], "a direct llm.invoke reached the network: %r" % attempts


def test_the_blocked_client_invoke_methods_raise_the_guard_message():
    """The mechanism, pinned directly: calling the boto3 client's socket-opening methods
    (converse / converse_stream / invoke_model) raises the guard's own RuntimeError -- so the error
    a developer sees names the fix, and no socket is opened."""
    _require_live_bedrock_llm()
    client = cfg.llm.client
    for method in ("converse", "converse_stream", "invoke_model"):
        with _socket_spy() as attempts:
            with pytest.raises(RuntimeError) as excinfo:
                getattr(client, method)(modelId="x", messages=[])
        assert "blocked" in str(excinfo.value), excinfo.value
        assert attempts == [], "%s reached the network: %r" % (method, attempts)


def test_deliberate_regression_an_unblocked_real_client_attempts_the_network():
    """THE REGRESSION, kept permanently: build a REAL llm through the live init_llm path with the
    provider/model/temperature app.config itself resolved (read live, so a prior reload cannot hand
    a stale value), then drive invoke under the spy. It MUST attempt the network -- proving the tests
    above are not green by accident: the guard is the only thing between the suite and a socket. (The
    spy refuses the attempt, so nothing leaves the host and nothing is billed.)"""
    real_llm = cfg.init_llm(cfg.LLM_PROVIDER, cfg.LLM_MODEL, cfg.LLM_TEMPERATURE)
    assert real_llm is not None, "init_llm returned None -- LLM_PROVIDER is not a socket-opening provider"
    with _socket_spy() as attempts:
        with pytest.raises(Exception):
            real_llm.invoke("hello")
    assert attempts, "an unblocked real client made NO network attempt -- the spy or the client is not real"
    assert any(kind in ("getaddrinfo", "connect") for kind, *_ in attempts), attempts
    # N2: under bedrock the host is the provider endpoint, not IMDS -- confirm the attempt is NOT the
    # metadata service (that would mean creds were never supplied, a different defect).
    assert not any(_IMDS_IP in str(a) for a in attempts), attempts


def test_a_stubbed_llm_is_left_untouched(monkeypatch):
    """NO COLLATERAL DAMAGE: the guard must not touch a holder that a test has stubbed with its own
    object (no boto3 `.client`). Replacing both live holders with a plain stub, the guard finds
    nothing to block and does not error -- the ~suites that stub `llm`/summarize_files are unaffected."""

    class _StubLLM:
        def invoke(self, *_a, **_k):
            return "stub summary"

    stub = _StubLLM()
    monkeypatch.setattr(cfg, "llm", stub)
    monkeypatch.setattr(document_routes, "llm", stub)
    assert _live_bedrock_llms() == [], "a stub with no .client was treated as a live bedrock holder"
    _guard_every_live_llm_client(monkeypatch)  # must not raise
    assert cfg.llm is stub and cfg.llm.invoke() == "stub summary"


def test_the_guard_pins_the_client_attribute_name():
    """The guard patches ChatBedrockConverse.client with raising=True, so a rename of that attribute
    ERRORS at setup instead of silently un-guarding. Name the contract here so the failure is
    readable: this is the attribute the guard relies on."""
    _require_live_bedrock_llm()
    assert hasattr(cfg.llm, "client")


def test_the_guard_covers_an_llm_rebuilt_by_a_config_reload(monkeypatch):
    """THE RELOAD GAP (N1 test-6 lesson applied to the LLM). test_batch_processing reloads app.config;
    a guard bound to the import-time object would leave the rebuilt llm's real client reachable.
    Reproduce the reload, confirm the fresh client really is unguarded (so this cannot pass
    vacuously), re-run the guard exactly as every test's setup does, and prove the NEW live client is
    blocked with zero socket attempts."""
    old = cfg.llm
    reload(cfg)
    assert cfg.llm is not old, "reload did not rebuild llm -- nothing to test"
    fresh_client = cfg.llm.client
    assert not isinstance(fresh_client, _BlockedBedrockRuntimeClient), "fresh llm was already guarded?"

    _guard_every_live_llm_client(monkeypatch)  # == the next test's setup
    with _socket_spy() as attempts:
        with pytest.raises(RuntimeError) as excinfo:
            cfg.llm.client.converse(modelId="x", messages=[])
    assert "blocked" in str(excinfo.value), excinfo.value
    assert attempts == [], "the REBUILT llm reached the network: %r" % attempts


def test_the_summarizer_gate_is_pinned_off_and_flipping_it_still_cannot_reach_a_provider():
    """PIN THE GATE (N5 deliverable 1). SUM_UP_KNOWLEDGE_FILES is False under the test env, so paid
    summarization is off by default. But a default is not a guard: even if the gate were flipped, the
    code it gates (summarize_file_chunks) must STILL be unable to reach a provider -- because the
    client guard, not the gate, is what stands between the suite and inference. Prove both here."""
    assert cfg.SUM_UP_KNOWLEDGE_FILES is False, cfg.SUM_UP_KNOWLEDGE_FILES
    assert document_routes.SUM_UP_KNOWLEDGE_FILES is False, document_routes.SUM_UP_KNOWLEDGE_FILES
    _require_live_bedrock_llm()
    # gate flipped for this test -- the summarizer code would now run -- and it is still blocked.
    with _socket_spy() as attempts:
        with pytest.raises(Exception):
            summarize_file_chunks(cfg.llm, [Document(page_content="paragraph")])
    assert attempts == [], "the gated summarizer path reached the network: %r" % attempts


def test_the_config_import_makes_no_imds_probe():
    """IMDS (N5 deliverable 3). Re-import app.config under the socket spy and assert ZERO attempts to
    the EC2 metadata service (169.254.169.254). With AWS_EC2_METADATA_DISABLED=true and explicit dummy
    creds+region set in conftest before the first import, botocore never consults IMDS when it builds
    the bedrock client. (Other hosts -- e.g. a Redis probe -- may be attempted and refused; only the
    metadata service is asserted absent, which is the import-time hazard this pins.)"""
    with _socket_spy() as attempts:
        reload(cfg)
    imds = [a for a in attempts if _IMDS_IP in str(a)]
    assert imds == [], "app.config import probed the EC2 metadata service: %r" % imds
    # NOTE: cfg is left reloaded (as N1 test-6 does); the autouse guard re-walks the live holders at
    # the next test's setup, so the freshly rebuilt llm is re-blocked before any later test runs.

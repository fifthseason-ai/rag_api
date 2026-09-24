"""The test suite cannot reach the real EMBEDDINGS provider -- on either path, on every live holder.

N1 (2026-09-24), the embeddings twin of tests/test_rerank_provider_is_blocked_in_tests.py.
MEASURED: conftest sets EMBEDDINGS_PROVIDER=openai plus a dummy key, so app.config builds a LIVE
CachingEmbeddings(OpenAIEmbeddings(dummy)). RedisCache swallows every error, so with no Redis a
test is a cache MISS and BOTH paths reach the real provider: embed_query (the /query path via
document_routes.get_cached_query_embedding) and embed_documents (ingestion). With a dummy key the
call is rejected before inference, so it is not billed -- but "unbilled because the key is wrong"
is not the standard. A suite must not be ABLE to open a socket to a paid provider.

The guard is the autouse fixture `_block_the_real_embeddings_provider` in tests/conftest.py. This
file is its CONTROL, run with a SOCKET SPY: `socket.getaddrinfo` and `socket.socket.connect` are
patched to record and refuse, so "no attempt" is asserted at the OS boundary, not inferred. That
is the in-suite, permanent stand-in for `--network none` (the reviewer reproduces under a real
--network none as well; the spy records at getaddrinfo, which fires even when DNS is dead).

LIVE RESOLUTION, everywhere: tests/test_batch_processing.py reloads app.config in-process and
restores only the env var, so after it app.config.vector_store is a NEW object (new real inner
client) while document_routes keeps the OLD one, and app.config's EmbeddingsProvider is a NEW
enum class. So nothing here binds a config object at import: every test reads `app.config` at
call time, and the guard walks every live holder at setup. Test 6 pins that gap (measured in CI
run 36001855369, where a guard bound at import left the rebuilt object reachable).

The Redis probe: CachingEmbeddings consults the cache BEFORE the provider, and that probe is itself
a socket attempt (to REDIS_HOST). These tests stub the cache on EVERY live holder so the spy
isolates the PROVIDER path exactly -- a Redis probe must neither mask nor mimic a provider attempt.

Controls:
  * remove the guard's setattr in conftest      -> tests 1, 2 and 6 red (spy records an attempt)
  * test 3 is the PERMANENT deliberate-regression: with the guard bypassed for one test, a real
    client DOES attempt the network -- so the guard is what stands between the suite and a socket.
"""
import socket
from contextlib import contextmanager
from importlib import reload

import pytest

import app.config as cfg
from app.routes import document_routes
from app.services.cache import CachingEmbeddings
from tests.conftest import _guard_every_live_embeddings_provider, _live_caching_embeddings


class _NoCache:
    """A cache that is always a miss and never talks to Redis (isolates the provider path)."""

    def get(self, key):
        return None

    def set(self, key, value):
        return None


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


@pytest.fixture
def _no_redis_probe(monkeypatch):
    """Stub the cache on EVERY live CachingEmbeddings (routes' copy and app.config's may differ
    after a reload), and return the live app.config one."""
    live = _live_caching_embeddings()
    assert live, "no live CachingEmbeddings found -- the guard has nothing to stand on"
    for ce in live:
        monkeypatch.setattr(ce, "_cache", _NoCache())
    return cfg.vector_store.embedding_function


def test_the_query_path_is_blocked_before_any_socket(_no_redis_probe):
    """PATH 1 -- /query: document_routes.get_cached_query_embedding is the real entry the routes
    call. With the guard on it must fail with the guard's own message and open NO socket."""
    with _socket_spy() as attempts:
        with pytest.raises(RuntimeError) as excinfo:
            document_routes.get_cached_query_embedding("what is in the deck")
    assert "blocked" in str(excinfo.value), excinfo.value
    assert attempts == [], "the query path reached the network: %r" % attempts


def test_the_ingestion_path_is_blocked_before_any_socket(_no_redis_probe):
    """PATH 2 -- ingestion: embed_documents on the live embedding_function (what aadd_documents
    drives). Same guard, same outcome: its own message, NO socket."""
    with _socket_spy() as attempts:
        with pytest.raises(RuntimeError) as excinfo:
            cfg.vector_store.embedding_function.embed_documents(["chunk one", "chunk two"])
    assert "blocked" in str(excinfo.value), excinfo.value
    assert attempts == [], "the ingestion path reached the network: %r" % attempts


def test_deliberate_regression_an_unblocked_real_client_attempts_the_network(_no_redis_probe, monkeypatch):
    """THE REGRESSION, kept permanently: bypass the guard for THIS test by putting a REAL provider
    client -- built through the real init_embeddings path with the provider and model app.config
    itself resolved (read live, so a prior reload cannot hand us a stale enum) -- back as the inner
    client, then drive the same path under the spy. It MUST attempt the network. This proves the
    tests above are not green by accident: the guard is the only thing between the suite and a
    socket. (The spy refuses the attempt, so nothing leaves the host and nothing is billed.)"""
    real_client = cfg.init_embeddings(cfg.EMBEDDINGS_PROVIDER, cfg.EMBEDDINGS_MODEL)
    monkeypatch.setattr(_no_redis_probe, "_embeddings", real_client)

    with _socket_spy() as attempts:
        with pytest.raises(Exception):
            cfg.vector_store.embedding_function.embed_query("what is in the deck")

    assert attempts, "an unblocked real client made NO network attempt -- the spy or the client is not real"
    assert any(kind in ("getaddrinfo", "connect") for kind, *_ in attempts), attempts


def test_a_test_that_stubs_embedding_function_is_unaffected(monkeypatch):
    """NO COLLATERAL DAMAGE: the guard must not touch the suites that stub embedding_function
    at the top (test_main and friends). Their stub is used, and nothing goes near a socket."""

    class _Stub:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(document_routes.vector_store, "embedding_function", _Stub())
    with _socket_spy() as attempts:
        out = document_routes.get_cached_query_embedding("q")
    assert out == [0.1, 0.2, 0.3], out
    assert attempts == [], attempts


def test_the_guard_pins_the_inner_attribute_name():
    """The guard patches CachingEmbeddings._embeddings with raising=True, so a rename of that
    attribute ERRORS at setup instead of silently un-guarding. Name the contract here so the
    failure is readable: this is the attribute the guard relies on."""
    assert hasattr(CachingEmbeddings(object(), _NoCache()), "_embeddings")


def test_the_guard_covers_a_vector_store_rebuilt_by_a_config_reload(monkeypatch):
    """THE MEASURED GAP (CI run 36001855369 on 275c263). test_batch_processing reloads app.config
    and restores only the env var, so app.config.vector_store becomes a NEW object with a NEW real
    inner client; a guard bound to the import-time object left it reachable for the rest of the
    session. Reproduce the reload, confirm the fresh object really is unguarded (so this cannot pass
    vacuously), re-run the guard exactly as every test's setup does, and prove the NEW live object
    is blocked with zero socket attempts."""
    old = cfg.vector_store
    reload(cfg)
    assert cfg.vector_store is not old, "reload did not rebuild vector_store -- nothing to test"
    fresh_inner = cfg.vector_store.embedding_function._embeddings
    assert type(fresh_inner).__name__ != "_BlockedProviderEmbeddings", "fresh object was already guarded?"

    _guard_every_live_embeddings_provider(monkeypatch)          # == the next test's setup
    for ce in _live_caching_embeddings():
        monkeypatch.setattr(ce, "_cache", _NoCache())

    with _socket_spy() as attempts:
        with pytest.raises(RuntimeError) as excinfo:
            cfg.vector_store.embedding_function.embed_query("q")
    assert "blocked" in str(excinfo.value), excinfo.value
    assert attempts == [], "the REBUILT vector_store reached the network: %r" % attempts

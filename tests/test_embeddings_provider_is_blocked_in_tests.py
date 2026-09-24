"""The test suite cannot reach the real EMBEDDINGS provider -- on either path.

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

The Redis probe: CachingEmbeddings consults the cache BEFORE the provider, and that probe is itself
a socket attempt (to REDIS_HOST). These tests stub the cache to a no-op so the spy isolates the
PROVIDER path exactly -- the property under test is "the provider is unreachable", and a Redis
probe must neither mask a provider attempt nor be mistaken for one.

Controls:
  * remove the guard's setattr in conftest      -> tests 1 and 2 red (spy records an attempt)
  * test 3 is the PERMANENT deliberate-regression: with the guard bypassed for one test, a real
    client DOES attempt the network -- so the guard is what stands between the suite and a socket.
"""
import socket
from contextlib import contextmanager

import pytest

from app.config import EMBEDDINGS_MODEL, EmbeddingsProvider, init_embeddings, vector_store
from app.routes import document_routes
from app.services.cache import CachingEmbeddings


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
    ef = vector_store.embedding_function
    assert isinstance(ef, CachingEmbeddings), type(ef)
    monkeypatch.setattr(ef, "_cache", _NoCache())
    return ef


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
            vector_store.embedding_function.embed_documents(["chunk one", "chunk two"])
    assert "blocked" in str(excinfo.value), excinfo.value
    assert attempts == [], "the ingestion path reached the network: %r" % attempts


def test_deliberate_regression_an_unblocked_real_client_attempts_the_network(_no_redis_probe, monkeypatch):
    """THE REGRESSION, kept permanently: bypass the guard for THIS test by putting a REAL provider
    client (built through the real init_embeddings path, not a hand-made stub) back as the inner
    client, then drive the same path under the spy. It MUST attempt the network. This proves the
    two tests above are not green by accident: the guard is the only thing between the suite and
    a socket. (The spy refuses the attempt, so nothing leaves the host and nothing is billed.)"""
    real_client = init_embeddings(EmbeddingsProvider.OPENAI, EMBEDDINGS_MODEL)
    monkeypatch.setattr(_no_redis_probe, "_embeddings", real_client)

    with _socket_spy() as attempts:
        with pytest.raises(Exception):
            vector_store.embedding_function.embed_query("what is in the deck")

    assert attempts, "an unblocked real client made NO network attempt -- the spy or the client is not real"
    assert any(kind == "getaddrinfo" or kind == "connect" for kind, *_ in attempts), attempts


def test_a_test_that_stubs_embedding_function_is_unaffected(monkeypatch):
    """NO COLLATERAL DAMAGE: the guard must not touch the suites that stub embedding_function
    at the top (test_main and friends). Their stub is used, and nothing goes near a socket."""

    class _Stub:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(vector_store, "embedding_function", _Stub())
    with _socket_spy() as attempts:
        out = document_routes.get_cached_query_embedding("q")
    assert out == [0.1, 0.2, 0.3], out
    assert attempts == [], attempts


def test_the_guard_pins_the_inner_attribute_name():
    """The guard patches CachingEmbeddings._embeddings with raising=True, so a rename of that
    attribute ERRORS at setup instead of silently un-guarding. Name the contract here so the
    failure is readable: this is the attribute the guard relies on."""
    assert hasattr(CachingEmbeddings(object(), _NoCache()), "_embeddings")

"""F-EMBED-CACHE: the embedding cache (CachingEmbeddings + RedisCache) wraps EVERY
/embed and /query embedding by default and had zero test coverage (F-COVERAGE-SWEEP).

Proven here, each against a fixture that can express the failure:
  * partial cache hits in embed_documents keep INPUT order (vectors differ per text, and
    the hits sit between misses, so any reassembly slip moves a vector);
  * the key namespace is per model: the same text under two models never cross-hits;
  * Redis down (connection refused, real redis client) -> vectors still computed, a
    warning logged, never an error and never an empty vector;
  * MEASURED DEFECT 1: an UNREACHABLE Redis (every call waits out socket_timeout) made
    5 embeddings take 20 s on main; after the first failure the cache is now bypassed
    for a cooldown, so the provider is called without further Redis waits;
  * MEASURED DEFECT 2: a cached value that is not JSON raised JSONDecodeError out of
    embed_query on main (a /query 500); it is now a miss;
  * a cached empty vector is a miss, never returned.
No Redis server and no provider are needed: the provider is a deterministic fake and
the Redis client is either real-but-refused or a scripted fake.
"""
import logging
import time

import pytest

from app.services.cache.embeddings import CachingEmbeddings
from app.services.cache.redis_cache import RedisCache


def _vec(model, text):
    # Distinct per (model, text): a vector in the wrong slot or from the wrong model shows.
    return [float(len(text)), float(sum(map(ord, text)) % 997), float(len(model))]


class _Provider:
    def __init__(self, model="model-a"):
        self.model = model
        self.query_calls = []
        self.doc_calls = []

    def embed_query(self, text):
        self.query_calls.append(text)
        return _vec(self.model, text)

    def embed_documents(self, texts):
        self.doc_calls.append(list(texts))
        return [_vec(self.model, t) for t in texts]


class _DictCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def evict(self, key):
        self.store.pop(key, None)


class _ScriptedClient:
    """Stands in for redis.Redis. `fail` raises on every call; `raw` is what get returns."""

    def __init__(self, fail=None, raw=None, delay=0.0):
        self.fail, self.raw, self.delay = fail, raw, delay
        self.calls = 0

    def _op(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail

    def get(self, key):
        self._op()
        return self.raw

    def set(self, key, value, ex=None):
        self._op()

    def delete(self, key):
        self._op()
        return 1


def _redis(client=None, **kw):
    cache = RedisCache(**kw)
    if client is not None:
        cache._client = client
    return cache


def test_partial_hits_keep_input_order():
    cache = _DictCache()
    provider = _Provider()
    emb = CachingEmbeddings(provider, cache, namespace="model-a")
    texts = ["alpha", "bravo-bravo", "c", "delta delta delta", "echo!"]

    # Warm ONLY the 2nd and 4th, so hits sit between misses.
    emb.embed_documents([texts[1], texts[3]])
    provider.doc_calls.clear()

    got = emb.embed_documents(texts)

    assert provider.doc_calls == [["alpha", "c", "echo!"]], "only the misses are computed"
    assert got == [_vec("model-a", t) for t in texts]


def test_namespace_is_per_model():
    cache = _DictCache()
    a = CachingEmbeddings(_Provider("model-a"), cache, namespace="model-a")
    b_provider = _Provider("model-bb")
    b = CachingEmbeddings(b_provider, cache, namespace="model-bb")

    assert a.embed_query("same text") == _vec("model-a", "same text")
    assert b.embed_query("same text") == _vec("model-bb", "same text")
    assert b_provider.query_calls == ["same text"], "model B must not hit model A's entry"
    assert b.embed_documents(["same text"]) == [_vec("model-bb", "same text")]


def test_redis_refused_still_embeds_and_warns(caplog):
    # Real redis client, nothing listening on port 1: a genuine connection error.
    cache = _redis(host="127.0.0.1", port=1, socket_timeout=0.5, failure_cooldown=0.0)
    provider = _Provider()
    emb = CachingEmbeddings(provider, cache, namespace="model-a")

    with caplog.at_level(logging.WARNING):
        q = emb.embed_query("hello")
        docs = emb.embed_documents(["one", "two"])

    assert q == _vec("model-a", "hello")
    assert docs == [_vec("model-a", "one"), _vec("model-a", "two")]
    assert all(v for v in [q, *docs]), "never an empty vector"
    assert any("Redis get failed" in r.getMessage() for r in caplog.records)


def test_unreachable_redis_is_bypassed_after_first_failure():
    """MEASURED on main: 5 docs against a blackholed host took 20.02 s (2 s x get+set x 5).
    Each client call here costs 0.2 s; after the first failure no further call is made."""
    client = _ScriptedClient(fail=TimeoutError("timed out"), delay=0.2)
    cache = _redis(client, failure_cooldown=60.0)
    emb = CachingEmbeddings(_Provider(), cache, namespace="model-a")

    t0 = time.monotonic()
    got = emb.embed_documents([f"t{i}" for i in range(5)])
    elapsed = time.monotonic() - t0

    assert got == [_vec("model-a", f"t{i}") for i in range(5)]
    assert client.calls == 1, client.calls
    assert elapsed < 1.0, elapsed
    # Still bypassed for the next request inside the cooldown.
    emb.embed_query("again")
    assert client.calls == 1


def test_cooldown_expiry_retries_redis():
    client = _ScriptedClient(fail=ConnectionError("down"))
    cache = _redis(client, failure_cooldown=0.05)
    assert cache.get("k") is None and client.calls == 1
    assert cache.get("k") is None and client.calls == 1  # bypassed
    time.sleep(0.1)
    client.fail, client.raw = None, "[1.0, 2.0]"
    assert cache.get("k") == [1.0, 2.0] and client.calls == 2


def test_evict_is_never_bypassed():
    client = _ScriptedClient(fail=ConnectionError("down"))
    cache = _redis(client, failure_cooldown=60.0)
    cache.get("k")
    client.fail = None
    cache.evict("k")
    assert client.calls == 2


@pytest.mark.parametrize("raw", ["{not json", "", "\x00\x01"])
def test_corrupt_cached_value_is_a_miss(raw, caplog):
    """MEASURED on main: '{not json' raised JSONDecodeError out of embed_query."""
    client = _ScriptedClient(raw=raw)
    provider = _Provider()
    emb = CachingEmbeddings(provider, _redis(client), namespace="model-a")
    with caplog.at_level(logging.WARNING):
        assert emb.embed_query("x") == _vec("model-a", "x")
        assert emb.embed_documents(["y"]) == [_vec("model-a", "y")]
    assert provider.query_calls == ["x"] and provider.doc_calls == [["y"]]


@pytest.mark.parametrize("raw", ["[]", "null", '"text"', "{}"])
def test_cached_non_vector_is_a_miss(raw):
    provider = _Provider()
    emb = CachingEmbeddings(provider, _redis(_ScriptedClient(raw=raw)), namespace="model-a")
    assert emb.embed_query("x") == _vec("model-a", "x")
    assert emb.embed_documents(["y"]) == [_vec("model-a", "y")]


def test_real_redis_round_trip_value_is_the_vector():
    """A hit returns exactly what was stored (JSON round trip through the real code)."""
    stored = {}

    class _KV(_ScriptedClient):
        def get(self, key):
            return stored.get(key)

        def set(self, key, value, ex=None):
            stored[key] = value

    provider = _Provider()
    emb = CachingEmbeddings(provider, _redis(_KV()), namespace="model-a")
    first = emb.embed_query("round trip")
    second = emb.embed_query("round trip")
    assert first == second == _vec("model-a", "round trip")
    assert provider.query_calls == ["round trip"], "second call must be a cache hit"

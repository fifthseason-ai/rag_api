"""Every retrieval leg must impose a TOTAL order, so the same question gives the same answer.

F-QUERY-TIE-ORDER-NOT-TOTAL (PACKET 1 finding, pre-existing at 5816e133). Bears directly on
Richard's ask-twice-same-answer requirement. Measured by PACKET 1 under a forced parallel plan:
dense `/query` k=4 returned **11 distinct result sets in 25 repeats**; a duplicated DOCX cited
fA or fB between repeats; serial was stable; reversing insert order changed the order again --
i.e. no row key decided the tie, the executor did.

All three legs sorted on a NON-UNIQUE key:
  * dense   -- langchain_community 0.4.1 pgvector.py `_query_collection`: ORDER BY distance ONLY
  * keyword -- hybrid_search.py: ORDER BY score DESC ONLY
  * rerank  -- reranker.rerank: kept the provider's sequence for equal relevanceScore

Why it cannot be fixed in the caller: both SQL legs apply `LIMIT k` IN THE DATABASE, so a tie at
the k-boundary decides WHICH ROWS COME BACK AT ALL. Sorting afterwards cannot recover a row the
query never returned. The tiebreaker has to be in the ORDER BY.

These tests are HERMETIC -- they capture the constructed query/SQL through fakes; no database, no
network, no provider call. The real-pg tie tests (including the parallel-plan one, which needs a
`Workers Launched > 0` precondition and therefore the heavy slot) are the other half of the
acceptance and are NOT in this file.

Controls:
  * drop `asc(EmbeddingStore.uuid)` from the dense override      -> the dense test reds
  * restore `ORDER BY score DESC` (no uuid) in hybrid_search     -> the keyword test reds
  * remove the explicit sort in reranker.rerank                  -> the rerank tie test reds
  * let a langchain upgrade replace our `_query_collection`      -> the override-identity test reds
"""
import pytest
from langchain_core.documents import Document
from sqlalchemy import column

import app.services.hybrid_search as hybrid_search
import app.services.reranker as reranker
import app.services.vector_store.extended_pg_vector as epv
from app.services.vector_store.extended_pg_vector import ExtendedPgVector


# --- dense leg -------------------------------------------------------------------------


class _RecordingSession:
    """Records the chained query construction. Touches no database."""

    def __init__(self):
        self.order_by_args = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def order_by(self, *args):
        self.order_by_args = args
        return self

    def join(self, *a, **k):
        return self

    def limit(self, *a):
        return self

    def all(self):
        return []


class _Distance:
    def label(self, name):
        return name  # upstream does distance_strategy(emb).label("distance")


def _store_without_a_database(monkeypatch, recorder):
    """A real ExtendedPgVector built WITHOUT __init__ (which would connect). The class's
    own __del__ is already guarded for exactly this shape, so collection is safe."""
    store = object.__new__(ExtendedPgVector)
    store._bind = object()
    store.use_jsonb = True
    store.EmbeddingStore = type(
        "EmbeddingStore", (), {"collection_id": column("collection_id"), "uuid": column("uuid")}
    )
    store.CollectionStore = type("CollectionStore", (), {"uuid": column("collection_uuid")})
    store.get_collection = lambda session: type("Collection", (), {"uuid": "coll-1"})
    # `distance_strategy` is a read-only property upstream, so patch it on the class.
    monkeypatch.setattr(
        ExtendedPgVector,
        "distance_strategy",
        property(lambda self: (lambda embedding: _Distance())),
        raising=False,
    )
    monkeypatch.setattr(epv, "Session", lambda bind: recorder)
    return store


def test_dense_leg_orders_by_distance_then_row_uuid(monkeypatch):
    """THE FIX, dense leg: distance alone is a partial order; uuid makes it total."""
    recorder = _RecordingSession()
    store = _store_without_a_database(monkeypatch, recorder)

    store._query_collection([0.1, 0.2, 0.3, 0.4], k=3, filter=None)

    assert recorder.order_by_args is not None, "the query was never ordered"
    rendered = [str(clause) for clause in recorder.order_by_args]
    assert len(rendered) == 2, rendered
    assert rendered[0] == "distance ASC", rendered
    assert "uuid" in rendered[1] and "ASC" in rendered[1], rendered


def test_the_dense_override_is_ours_not_langchains():
    """If a langchain upgrade replaces the method we copied, the total order silently
    reverts to upstream's partial one. Pin the override's identity so that fails loudly."""
    from langchain_community.vectorstores.pgvector import PGVector

    assert ExtendedPgVector._query_collection is not PGVector._query_collection


# --- keyword leg -----------------------------------------------------------------------


class _RecordingConnection:
    def __init__(self, sink):
        self.sink = sink

    async def fetch(self, sql, *params):
        self.sink["sql"] = sql
        self.sink["params"] = params
        return []


class _Acquire:
    def __init__(self, sink):
        self.sink = sink

    async def __aenter__(self):
        return _RecordingConnection(self.sink)

    async def __aexit__(self, *exc):
        return False


class _RecordingPool:
    def __init__(self, sink):
        self.sink = sink

    def acquire(self):
        return _Acquire(self.sink)


async def test_keyword_leg_orders_by_score_then_row_uuid(monkeypatch):
    """THE FIX, keyword leg. ts_rank_cd ties are common, so score DESC alone is partial."""
    sink = {}

    async def _get_pool():
        return _RecordingPool(sink)

    monkeypatch.setattr(hybrid_search.PSQLDatabase, "get_pool", staticmethod(_get_pool))

    await hybrid_search.keyword_search("alpha beta", k=3, filters={"user_id": "uA"})

    sql = " ".join(sink["sql"].split())
    assert "ORDER BY score DESC, uuid ASC" in sql, sql


# --- rerank leg ------------------------------------------------------------------------


async def test_rerank_breaks_relevance_ties_by_candidate_index(monkeypatch):
    """THE FIX, rerank leg: the provider's sequence is not a guaranteed order for equal
    relevanceScore, so impose one. Ties resolve by the candidate's own position."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    candidates = [(Document(page_content=t), 0.1) for t in ("a", "b", "c")]

    # Two hits TIE at 0.5 and the provider returns the later candidate first.
    monkeypatch.setattr(
        reranker,
        "_rerank_sync",
        lambda q, docs, n: [
            {"index": 2, "relevanceScore": 0.5},
            {"index": 0, "relevanceScore": 0.5},
            {"index": 1, "relevanceScore": 0.9},
        ],
    )

    out = await reranker.rerank("q", candidates, top_n=3)

    # 0.9 leads; the 0.5 tie resolves by candidate index (0 before 2), NOT provider order.
    assert [doc.page_content for doc, _score in out] == ["b", "a", "c"], out
    assert [round(score, 3) for _doc, score in out] == [0.9, 0.5, 0.5], out


async def test_rerank_keeps_a_strict_relevance_ordering(monkeypatch):
    """NO REGRESSION: distinct scores still order by relevance, highest first."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    candidates = [(Document(page_content=t), 0.1) for t in ("a", "b", "c")]
    monkeypatch.setattr(
        reranker,
        "_rerank_sync",
        lambda q, docs, n: [
            {"index": 0, "relevanceScore": 0.2},
            {"index": 1, "relevanceScore": 0.8},
            {"index": 2, "relevanceScore": 0.5},
        ],
    )

    out = await reranker.rerank("q", candidates, top_n=3)
    assert [doc.page_content for doc, _score in out] == ["b", "c", "a"], out

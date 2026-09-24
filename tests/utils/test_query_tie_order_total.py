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
  * DELETE or RENAME our `_query_collection`                     -> the override-identity test reds
  * re-route the PUBLIC search off our override                  -> the public-path test reds
  * bump `langchain_community` off the pinned version            -> the version-pin test reds

UPGRADE SAFETY (was KNOWN LIMIT 1; RV-122 F1). The override-identity pin alone catches only
DELETION or RENAME. The reviewer measured two realistic upgrades that left it green: upstream
changing `_query_collection`'s body, and upstream routing the public search through a DIFFERENT
internal method (which would strand our override uncalled and silently restore the partial
order). Those two gaps are now closed from the other direction:
  * `test_the_public_search_path_still_flows_through_our_total_order` drives the PUBLIC
    `similarity_search_with_score_by_vector` and asserts it reaches our ordered code -- reds on
    a re-route;
  * `test_langchain_community_is_the_pinned_version` reds on a version bump, forcing a human to
    re-read upstream's body and re-confirm the copy.
The residual is honest and NAMED (RV-132): the version pin assumes one version string maps to
one body. That holds for an immutable PyPI release, but NOT for a PATCHED or VENDORED install that
keeps "0.4.1" while carrying a different `_query_collection` -- it changes the body without moving
the pin, and nothing here catches it. That case threatens copied-body FIDELITY, not tie order: the
public path still reaches our override, so the order stays total; only our copy could drift from a
locally-patched upstream. A source hash would catch it but re-reads noisily across identical
reinstalls; the version pin plus the public-path test is the proportionate guard for the ordinary
upgrade path, with the patched/vendored-install case stated rather than covered.

KNOWN LIMIT 2 -- the provider decides a tie we never see (RV-122 F2). `reranker._rerank_sync`
asks Bedrock for `numberOfResults: top_n` (reranker.py:65), so when relevance ties at the
k-boundary the PROVIDER chooses which candidates come back before our sort runs. The sort
below makes the returned set totally ordered; it cannot make the SET deterministic. This is
the same k-boundary argument this file makes about SQL `LIMIT`, applied one layer out, and
it is a real hole in the guarantee. Closing it means scoring the whole candidate pool and
cutting locally, whose cost effect is UNMEASURED -- and cannot be measured here, because
that measurement is itself a paid Bedrock call under the standing no-paid-calls hold. Stated
as a limit rather than guessed at.
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
    # RV-132 F1: our override uses epv.Session, but the UPSTREAM _query_collection body
    # references langchain_community...pgvector.Session. If an upgrade re-routes the public
    # search through the upstream body (the re-route the public-path test guards), that body
    # would hit the REAL Session and raise an opaque sqlalchemy ArgumentError -- a failure that
    # reads as a broken test, not as "the order reverted to single-key". Patch the upstream
    # module's Session too so a re-routed body records its (reverted) order and the public-path
    # test reds with its OWN assertion. Inert on the happy path: our override uses epv.Session.
    import langchain_community.vectorstores.pgvector as _lcpg
    monkeypatch.setattr(_lcpg, "Session", lambda bind: recorder, raising=False)
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
    """Catches DELETION or RENAME of our override -- and only that.

    It does NOT make a langchain upgrade fail loudly; see UPGRADE SAFETY in the module
    docstring. RV-122 measured two realistic upgrades that leave this green, including
    upstream routing the public search through a different internal method, which would
    strand our override uncalled and silently restore the partial order."""
    from langchain_community.vectorstores.pgvector import PGVector

    assert ExtendedPgVector._query_collection is not PGVector._query_collection


def test_the_public_search_path_still_flows_through_our_total_order(monkeypatch):
    """RV-122 F1, the gap the identity pin cannot see: `similarity_search_with_score_by_vector`
    is what the app actually calls. Pin that it reaches OUR ordered `_query_collection`. If a
    langchain upgrade re-routes the public method through a different internal one, our override
    is stranded and the order silently reverts -- and THIS reds, where the identity pin stays
    green. It asserts the call REACHES our code, not what the SQL says."""
    recorder = _RecordingSession()
    store = _store_without_a_database(monkeypatch, recorder)
    # _results_to_docs_and_scores runs after _query_collection; stub it so we exercise only
    # the routing, not row shaping.
    monkeypatch.setattr(
        ExtendedPgVector, "_results_to_docs_and_scores", lambda self, results: results,
        raising=False,
    )

    store.similarity_search_with_score_by_vector([0.1, 0.2, 0.3, 0.4], k=3)

    assert recorder.order_by_args is not None, (
        "the public search did not reach our ordered _query_collection -- upstream may have "
        "re-routed it (RV-122 F1)"
    )
    rendered = [str(c) for c in recorder.order_by_args]
    assert len(rendered) == 2 and "uuid" in rendered[1], rendered


def test_langchain_community_is_the_pinned_version():
    """RV-122 F1, the other half: our override copies upstream's `_query_collection` body, so
    an upstream body change is invisible to the tests above. Pin the version this override was
    verified against; a bump reds HERE, forcing a human to re-read upstream and re-confirm the
    copy (and move this pin forward) rather than discovering a partial order in production.
    A version string, not a source hash, on purpose: it is stable across reinstalls of the same
    release and names exactly what a reader must go check."""
    import langchain_community

    assert langchain_community.__version__ == "0.4.1", (
        "langchain_community moved from the version ExtendedPgVector._query_collection was "
        "copied from; re-read upstream _query_collection, re-confirm the override, bump this pin"
    )


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

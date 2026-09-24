"""What the number in a `/query*` hit MEANS, declared on the wire.

P06-5 retrieval contract, ADDENDUM 2 (FILES, 2026-09-23). The pair is `[document, score]` and
the score's meaning depends on which retrieval mode RAN: a pgvector dense search returns a
cosine DISTANCE (lower is better), the default hybrid search returns an RRF fused score and a
successful rerank returns the provider's relevance (both higher is better), and an Atlas store
returns MongoDB's vectorSearchScore (higher is better). Nothing on the wire said which, and the
consumer read every one of them as a distance -- so under the DEFAULT configuration it sorted
this service's best chunks last (PACKET 1 finding, program-control 5e49dfe).

Each hit's document object now carries `score_kind` and `score_direction`. The kind names the
mode that ACTUALLY ran -- a hybrid configuration that fell back to dense says
`cosine_distance`, a rerank that failed says the fallback's kind -- because it travels with the
result list from the point where the mode is decided (see `ScoredHits`), not from config.

A kind is declared only where it is KNOWN. A store this module cannot identify, or a list built
anywhere else (a test stub, a future caller that forgets), carries no kind, and the wire then
says `null`: UNKNOWN, which the consumer marks and never guesses from the value's range.
"""
from typing import Iterable, Optional

COSINE_DISTANCE = "cosine_distance"
RRF = "rrf"
RERANK_RELEVANCE = "rerank_relevance"
VECTOR_SEARCH_SCORE = "vector_search_score"

LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"

#: The closed set of kinds and the direction each one implies. The direction is stated on the
#: wire anyway so a consumer can order hits of a kind it does not recognise yet.
DIRECTION = {
    COSINE_DISTANCE: LOWER_IS_BETTER,
    RRF: HIGHER_IS_BETTER,
    RERANK_RELEVANCE: HIGHER_IS_BETTER,
    VECTOR_SEARCH_SCORE: HIGHER_IS_BETTER,
}


class ScoredHits(list):
    """A list of (Document, score) pairs that also says what its scores mean.

    It IS a list, so every existing caller, comparison and test that treats the result as a
    plain list keeps working. The kind rides along only as far as someone passes the object on:
    slicing or rebuilding produces a plain list, which is why every site that rebuilds a result
    re-wraps it with `kind_of(<its input>)` rather than letting the kind silently fall off.
    """

    def __init__(self, pairs: Iterable = (), score_kind: Optional[str] = None):
        super().__init__(pairs)
        if score_kind is not None and score_kind not in DIRECTION:
            raise ValueError("unknown score_kind %r; the closed set is %r" % (score_kind, sorted(DIRECTION)))
        self.score_kind = score_kind


def kind_of(pairs) -> Optional[str]:
    """The declared kind of a result list, or None (UNKNOWN) for anything undeclared."""
    return getattr(pairs, "score_kind", None)

# app/services/reranker.py
"""Rerank retrieval candidates with Cohere Rerank 3.5, served via AWS Bedrock (VI-438).

After hybrid search returns a candidate pool, a cross-encoder reads each
(query, chunk) pair jointly and reorders by true relevance. Best-effort: if
disabled or the call fails, candidates are returned in their original order.
"""
import asyncio
from typing import List, Tuple

import boto3
from langchain_core.documents import Document

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    RERANK_AWS_REGION,
    RERANK_ENABLED,
    RERANK_MODEL,
    logger,
)
from app.services.score_kind import RERANK_RELEVANCE, ScoredHits, kind_of

_client = None


def _get_client():
    """Cached Bedrock client (creating one is expensive). Mirrors the embeddings setup."""
    global _client
    if _client is None:
        session_kwargs = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "region_name": RERANK_AWS_REGION,
        }
        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
        _client = boto3.Session(**session_kwargs).client("bedrock-agent-runtime")
    return _client


def _rerank_sync(query: str, documents: List[str], top_n: int) -> list:
    """Blocking Bedrock Rerank API call (run in a thread to keep the loop free)."""
    model_arn = f"arn:aws:bedrock:{RERANK_AWS_REGION}::foundation-model/{RERANK_MODEL}"
    response = _get_client().rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=[
            {
                "type": "INLINE",
                "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": text}},
            }
            for text in documents
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": model_arn},
                # KNOWN LIMIT (RV-122 F2): asking for only top_n lets the PROVIDER resolve a
                # relevance tie at the k-boundary before our own sort runs, so the returned SET
                # is not deterministic on ties -- the same k-boundary problem the SQL legs have,
                # one layer out. Closing it means requesting the whole pool and cutting locally;
                # its cost is unmeasured and unmeasurable under the paid-call hold. See
                # tests/utils/test_query_tie_order_total.py "KNOWN LIMIT 2".
                "numberOfResults": top_n,
            },
        },
    )
    return response.get("results", [])


async def rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_n: int,
) -> List[Tuple[Document, float]]:
    """Rerank (Document, score) candidates, returning the top_n as
    (Document, relevance_score). Falls back to candidates[:top_n] on any failure
    or when disabled.

    The result declares what its scores mean (`app.services.score_kind`): a successful
    rerank is `rerank_relevance`; EVERY fallback keeps the candidates' own kind, because
    those numbers are still the candidates' numbers. A slice is a plain list, so each
    fallback re-wraps it -- otherwise the kind would silently fall off on exactly the
    paths (a failed Bedrock call, the default region) where it matters most."""
    kind = kind_of(candidates)
    if not candidates:
        return ScoredHits([], kind)

    top_n = max(1, min(top_n, len(candidates)))
    if not RERANK_ENABLED:
        return ScoredHits(candidates[:top_n], kind)

    documents = [doc.page_content for doc, _score in candidates]
    try:
        results = await asyncio.to_thread(_rerank_sync, query, documents, top_n)
    except Exception as exc:
        logger.warning("[rerank] failed; using pre-rerank order: %s", exc)
        return ScoredHits(candidates[:top_n], kind)

    # TOTAL order, not the provider's. Nothing documents that equal relevanceScores come
    # back in a stable sequence, so two identical requests could interleave tied hits
    # differently -- and RV-118 flagged the same thing independently: this function never
    # sorted, it inherited whatever order Bedrock returned, which is INFERRED best-first,
    # not verified. Sort explicitly: relevance DESC, then the candidate's own position in
    # the pool, which is unique and (with the retrieval legs below now totally ordered)
    # itself deterministic.
    _scored = [
        (r["index"], candidates[r["index"]][0], float(r.get("relevanceScore", 0.0)))
        for r in results
        if 0 <= r.get("index", -1) < len(candidates)
    ]
    _scored.sort(key=lambda t: (-t[2], t[0]))
    reranked = [(doc, score) for _idx, doc, score in _scored]
    if not reranked:
        return ScoredHits(candidates[:top_n], kind)

    logger.info(
        "[rerank] cohere via bedrock (%s) | %d candidates -> top %d",
        RERANK_MODEL, len(candidates), len(reranked),
    )
    return ScoredHits(reranked, RERANK_RELEVANCE)

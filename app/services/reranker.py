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
    or when disabled."""
    if not candidates:
        return []

    top_n = max(1, min(top_n, len(candidates)))
    if not RERANK_ENABLED:
        return candidates[:top_n]

    documents = [doc.page_content for doc, _score in candidates]
    try:
        results = await asyncio.to_thread(_rerank_sync, query, documents, top_n)
    except Exception as exc:
        logger.warning("[rerank] failed; using pre-rerank order: %s", exc)
        return candidates[:top_n]

    reranked = [
        (candidates[r["index"]][0], float(r.get("relevanceScore", 0.0)))
        for r in results
        if 0 <= r.get("index", -1) < len(candidates)
    ]
    if not reranked:
        return candidates[:top_n]

    logger.info(
        "[rerank] cohere via bedrock (%s) | %d candidates -> top %d",
        RERANK_MODEL, len(candidates), len(reranked),
    )
    return reranked

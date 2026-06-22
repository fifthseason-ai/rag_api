# app/services/hybrid_search.py
"""Keyword (BM25-style) full-text search and Reciprocal Rank Fusion (VI-436).

Runs Postgres full-text search over the `document_tsv` generated column on
`langchain_pg_embedding` (created by tempo migration 10081) in parallel with the
existing dense vector search, then fuses the two ranked lists with RRF.

Native Postgres FTS (tsvector + GIN + ts_rank_cd) is used as a pragmatic
BM25 approximation; it captures exact terms (brand names, financial figures,
acronyms, slide labels) that dense embeddings tend to miss.
"""
from typing import Any, Dict, List, Optional, Tuple

import json
import re

from langchain_core.documents import Document

from app.config import FTS_CONFIG, RRF_K, logger
from app.services.database import PSQLDatabase

# Tokens for the OR tsquery: alphanumeric runs only. This strips punctuation and
# tsquery operators, so the resulting string is safe to feed to to_tsquery as a
# bound parameter ("tok1 | tok2 | tok3").
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _build_or_tsquery(query: str) -> str:
    """Turn a free-text query into an OR tsquery string: 'a | b | c'.

    Uses OR (not AND) semantics so a chunk matches when it contains ANY of the
    query terms; ts_rank_cd then ranks by how many/how relevant the matches are.
    This keeps the keyword arm useful for conceptual, multi-word queries instead
    of requiring every term to co-occur in a single chunk. Tokens are lowercased
    and de-duplicated while preserving order.
    """
    seen = set()
    tokens = []
    for match in _TOKEN_RE.findall(query.lower()):
        if match not in seen:
            seen.add(match)
            tokens.append(match)
    return " | ".join(tokens)


async def keyword_search(
    query: str,
    k: int = 4,
    filters: Optional[Dict[str, Any]] = None,
    fts_config: str = FTS_CONFIG,
) -> List[Tuple[Document, float]]:
    """Full-text keyword search over langchain_pg_embedding.document_tsv.

    Returns (Document, ts_rank_cd score) tuples ordered best-first (higher score
    is better). Respects the same cmetadata equality filters as the dense path;
    a list value (e.g. multiple file_ids) is matched with ANY/IN semantics.
    """
    if not query or not query.strip():
        logger.info("[keyword_search] empty query, skipping keyword search")
        return []

    ts_query = _build_or_tsquery(query)
    if not ts_query:
        logger.info("[keyword_search] no usable tokens in query, skipping keyword search")
        return []

    filters = filters or {}

    # $1 = text search config, $2 = OR tsquery string built from the user query
    # (e.g. "page | object | pattern"); safe to pass to to_tsquery as a param.
    where_clauses = ["document_tsv @@ to_tsquery($1::regconfig, $2)"]
    params: List[Any] = [fts_config, ts_query]

    for key, value in filters.items():
        if value is None:
            continue
        # Keys are caller-controlled (e.g. "file_id", "user_id"), never raw user
        # input — but validate before interpolating into SQL just in case.
        if not key.isidentifier():
            raise ValueError(f"Invalid filter key: {key!r}")
        if isinstance(value, (list, tuple, set)):
            values = [str(v) for v in value]
            if not values:
                continue
            params.append(values)
            where_clauses.append(f"cmetadata->>'{key}' = ANY(${len(params)}::text[])")
        else:
            params.append(str(value))
            where_clauses.append(f"cmetadata->>'{key}' = ${len(params)}")

    params.append(k)
    limit_idx = len(params)

    sql = f"""
        SELECT document,
               cmetadata,
               ts_rank_cd(document_tsv, to_tsquery($1::regconfig, $2)) AS score
        FROM langchain_pg_embedding
        WHERE {' AND '.join(where_clauses)}
        ORDER BY score DESC
        LIMIT ${limit_idx}
    """

    logger.info(
        "[keyword_search] running FTS (OR) | config=%s | k=%d | filters=%s | query=%r | tsquery=%r",
        fts_config, k, filters, query, ts_query,
    )

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    results: List[Tuple[Document, float]] = []
    for row in rows:
        metadata = row["cmetadata"]
        # asyncpg returns jsonb as a string unless a codec is registered.
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        results.append(
            (
                Document(page_content=row["document"], metadata=metadata or {}),
                float(row["score"]),
            )
        )

    logger.info("[keyword_search] matched %d chunks", len(results))
    return results


def _fusion_key(document: Document) -> str:
    """Stable identity for a chunk so it can be matched across both result sets."""
    digest = (document.metadata or {}).get("digest")
    if digest:
        return f"digest:{digest}"
    return f"content:{hash(document.page_content)}"


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    k: int = 4,
    rrf_k: int = RRF_K,
) -> List[Tuple[Document, float]]:
    """Combine ranked result lists with Reciprocal Rank Fusion.

    RRF uses only each item's rank within its list (score = sum of
    1 / (rrf_k + rank)), making it robust to the incompatible scales of cosine
    distance (dense, lower=better) and ts_rank_cd (keyword, higher=better) —
    both inputs are expected ordered best-first. Returns (Document, fused_score)
    tuples sorted by fused score descending, truncated to k.
    """
    fused_scores: Dict[str, float] = {}
    documents: Dict[str, Document] = {}

    for results in result_lists:
        for rank, (document, _score) in enumerate(results):
            key = _fusion_key(document)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            documents.setdefault(key, document)

    fused = [(documents[key], score) for key, score in fused_scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)

    if len(result_lists) > 1:
        total_inputs = sum(len(results) for results in result_lists)
        unique = len(fused)
        # total_inputs - unique = chunks that appeared in more than one list.
        logger.info(
            "[rrf] fused %d lists | %d input items -> %d unique chunks "
            "(%d overlapping) | returning top %d",
            len(result_lists),
            total_inputs,
            unique,
            total_inputs - unique,
            min(k, unique),
        )
        logger.debug(
            "[rrf] fused scores (higher=better): %s",
            [round(score, 6) for _doc, score in fused[:k]],
        )
    return fused[:k]

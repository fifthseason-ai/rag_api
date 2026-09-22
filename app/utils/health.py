# app/utils/health.py
from app.config import HYBRID_SEARCH_ENABLED, VECTOR_DB_TYPE, VectorDBType
from app.services.database import pg_health_check
from app.services.hybrid_search import keyword_search_state
from app.services.mongo_client import mongo_health_check


async def is_health_ok():
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        return await pg_health_check()
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return await mongo_health_check()
    else:
        return True


async def keyword_search_health():
    """The keyword arm's state for /health (F-HYBRID-HEALTH). Informational: it never changes
    `status`, because a missing keyword arm leaves /query answering dense-only, not down."""
    if not HYBRID_SEARCH_ENABLED:
        return {"state": "disabled"}
    if VECTOR_DB_TYPE != VectorDBType.PGVECTOR:
        return {"state": "not_applicable"}
    return await keyword_search_state()

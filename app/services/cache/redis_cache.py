# app/services/cache/redis_cache.py
"""Redis cache backend.

The single cache implementation. Local and production differ only in
connection settings: locally it points at the Redis container on
``localhost:6379``; in production it points at the AWS ElastiCache (Redis)
endpoint. Both speak the same protocol, so the same client works for both.

Keys are hashed (SHA-256) under a configurable prefix, so arbitrary key objects
are supported as long as they are ``str``-convertible. Values are stored as
JSON and must therefore be JSON-serializable. Keys expire after ``ttl`` seconds
(0 disables expiry).

All Redis errors are swallowed and logged: ``get`` returns ``None`` and
``set``/``evict`` become no-ops on failure, so a cache outage degrades to a
cache miss instead of failing the caller.
"""

import json
import logging
import hashlib
from typing import Generic, Optional

from app.services.cache.base import Cache, K, V

logger = logging.getLogger(__name__)


class RedisCache(Cache[K, V], Generic[K, V]):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        ttl: int = 0,
        key_prefix: str = "rag:emb:",
        ssl: bool = False,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
    ):
        # Imported lazily so importing the package doesn't require redis until
        # the cache is actually instantiated.
        import redis

        self._ttl = ttl
        self._key_prefix = key_prefix
        self._client = redis.Redis(
            host=host,
            port=port,
            ssl=ssl,
            password=password or None,
            decode_responses=True,
            socket_connect_timeout=socket_timeout,
            socket_timeout=socket_timeout,
        )
        logger.info(
            "Initialized Redis cache [host=%s][port=%d][ssl=%s][ttl=%d]",
            host,
            port,
            ssl,
            ttl,
        )

    def _redis_key(self, key: K) -> str:
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return f"{self._key_prefix}{digest}"

    def get(self, key: K) -> Optional[V]:
        redis_key = self._redis_key(key)
        try:
            raw = self._client.get(redis_key)
        except Exception as exc:
            logger.warning("Redis get failed [key=%s][error=%s]", redis_key, exc)
            return None
        if raw is None:
            logger.debug("Redis cache miss [key=%s]", redis_key)
            return None
        logger.debug("Redis cache hit [key=%s]", redis_key)
        return json.loads(raw)

    def set(self, key: K, value: V) -> None:
        redis_key = self._redis_key(key)
        try:
            self._client.set(
                redis_key,
                json.dumps(value),
                ex=self._ttl or None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Redis set failed [key=%s][error=%s]", redis_key, exc)
            return
        logger.debug("Redis cache set [key=%s][ttl=%d]", redis_key, self._ttl)

    def evict(self, key: K) -> None:
        redis_key = self._redis_key(key)
        try:
            removed = self._client.delete(redis_key)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Redis evict failed [key=%s][error=%s]", redis_key, exc)
            return
        logger.debug("Redis cache evict [key=%s][removed=%d]", redis_key, removed)

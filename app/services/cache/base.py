# app/services/cache/base.py
"""Generic, backend-agnostic cache interface.

The cache is parameterized over a key type ``K`` and a value type ``V`` so it
can be reused for anything (embeddings, summaries, arbitrary lookups), not just
one payload shape. The Redis backend implements the three primitives below;
higher-level callers compose them.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Cache(ABC, Generic[K, V]):
    """Contract for a key/value cache with least-recently-used semantics."""

    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Return the value stored under ``key`` or ``None`` on a miss."""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: K, value: V) -> None:
        """Store ``value`` under ``key``."""
        raise NotImplementedError

    @abstractmethod
    def evict(self, key: K) -> None:
        """Remove ``key`` from the cache. A no-op if it is not present."""
        raise NotImplementedError

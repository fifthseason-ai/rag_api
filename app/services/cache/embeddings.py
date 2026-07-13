# app/services/cache/embeddings.py
"""Caching decorator for a LangChain ``Embeddings`` instance.

Wraps any ``Embeddings`` object and consults a :class:`Cache` before delegating
to the underlying provider. Because the wrapper implements the ``Embeddings``
interface, the same cache is applied everywhere embeddings are produced: query
embedding in the API endpoints (``embed_query``) and document embedding during
ingestion (``embed_documents``). The async variants inherited from the base
class run these sync methods in an executor, so they are cached too.

Cache keys are namespaced by the embeddings model so switching models never
returns stale vectors. Cache failures degrade to a miss (the backend swallows
its own errors), so the provider is always the source of truth on a miss.
"""

import logging
from typing import List

from langchain_core.embeddings import Embeddings

from app.services.cache.base import Cache

logger = logging.getLogger(__name__)


class CachingEmbeddings(Embeddings):
    def __init__(
        self,
        embeddings: Embeddings,
        cache: Cache[str, List[float]],
        namespace: str = "",
    ):
        self._embeddings = embeddings
        self._cache = cache
        self._namespace = namespace

    def _key(self, text: str) -> str:
        return f"{self._namespace}\x00{text}"

    def embed_query(self, text: str) -> List[float]:
        key = self._key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedding = self._embeddings.embed_query(text)
        self._cache.set(key, embedding)
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(self._key(text))
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            computed = self._embeddings.embed_documents(missing_texts)
            for j, index in enumerate(missing_indices):
                results[index] = computed[j]
                self._cache.set(self._key(missing_texts[j]), computed[j])

        logger.info(
            "Embeddings cache [total=%d][cached=%d][computed=%d]",
            len(texts),
            len(texts) - len(missing_texts),
            len(missing_texts),
        )
        return results

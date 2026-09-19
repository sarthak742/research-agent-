"""Async wrapper around Tavily web search.

Two problems with the original ``self.tavily.search(...)`` call:

1. ``tavily-python`` is a *synchronous* SDK, so calling it directly inside an
   async function blocks the event loop (same problem as the LLM call). We
   run it in a worker thread with ``asyncio.to_thread`` so the loop stays
   responsive. (If the SDK ever ships a native async client, this is the one
   place that changes.)

2. No retries and no caching. We add both: transient failures are retried
   with backoff, and identical queries within a TTL window reuse results so
   we don't pay Tavily twice for the same string.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from tavily import TavilyClient

from config import get_settings

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when a search fails after all retries."""


class _TTLCache:
    """Tiny in-process TTL cache. Not thread-shared state to worry about
    because all access happens on the event loop thread."""

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        hit = self._store.get(key)
        if hit is None:
            return None
        expires_at, value = hit
        if time.monotonic() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)


class SearchClient:
    def __init__(
        self,
        api_key: str | None = None,
        # ``search_fn`` injection lets tests supply a fake with no network.
        search_fn: Callable[..., dict] | None = None,
    ):
        settings = get_settings()
        if search_fn is not None:
            self._search_fn = search_fn
        else:
            client = TavilyClient(api_key=api_key or settings.tavily_api_key)
            self._search_fn = client.search
        self._max_retries = settings.search_max_retries
        self._base_delay = settings.retry_base_delay
        self._results_per_query = settings.results_per_query
        self._cache = _TTLCache(settings.search_cache_ttl_seconds)

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Return a list of {title, url, content} dicts for a query."""
        cached = self._cache.get(query)
        if cached is not None:
            logger.debug("search cache hit: %s", query)
            return cached

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                raw = await asyncio.to_thread(
                    self._search_fn,
                    query=query,
                    max_results=self._results_per_query,
                )
                results = raw.get("results", []) if isinstance(raw, dict) else []
                self._cache.set(query, results)
                return results
            except Exception as exc:  # Tavily raises plain exceptions; retry all.
                last_exc = exc
                delay = self._base_delay * (2 ** attempt)
                logger.warning(
                    "search failed (attempt %d/%d) for %r: %s. Retrying in %.1fs",
                    attempt + 1, self._max_retries, query, exc, delay,
                )
                await asyncio.sleep(delay)
        raise SearchError(
            f"search failed after {self._max_retries} retries for {query!r}"
        ) from last_exc

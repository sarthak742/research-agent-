"""Session storage.

The original kept sessions in a bare module-level dict that was never
cleaned up — every research run leaked memory forever, and because the dict
lives in one process it also breaks the moment you run more than one worker.

This version:

* hides storage behind an interface (``SessionStore``) so the in-memory
  backend can be swapped for Redis/DB without touching the rest of the code;
* expires sessions after a TTL and sweeps them periodically, so memory is
  bounded;
* guards access with an ``asyncio.Lock`` so concurrent requests can't corrupt
  the dict.

Sessions are intentionally *ephemeral* — a research run is transient state,
not a record we need to keep — so an in-memory store with TTL is the honest
right default. The interface is the seam where you'd add Redis if you needed
to scale to multiple workers.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from agent.models import Session
from config import get_settings


class SessionStore(ABC):
    @abstractmethod
    async def create(self, session: Session) -> None: ...

    @abstractmethod
    async def get(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def save(self, session: Session) -> None: ...


class InMemorySessionStore(SessionStore):
    def __init__(self, ttl: float | None = None):
        s = get_settings()
        self._ttl = ttl if ttl is not None else s.session_ttl_seconds
        self._data: dict[str, tuple[float, Session]] = {}
        self._lock = asyncio.Lock()

    async def create(self, session: Session) -> None:
        async with self._lock:
            self._data[session.id] = (time.monotonic(), session)

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            hit = self._data.get(session_id)
            if hit is None:
                return None
            created, session = hit
            if time.monotonic() - created > self._ttl:
                self._data.pop(session_id, None)
                return None
            return session

    async def save(self, session: Session) -> None:
        async with self._lock:
            existing = self._data.get(session.id)
            created = existing[0] if existing else time.monotonic()
            self._data[session.id] = (created, session)

    async def sweep(self) -> int:
        """Drop expired sessions. Returns how many were removed."""
        now = time.monotonic()
        async with self._lock:
            expired = [
                sid for sid, (created, _) in self._data.items()
                if now - created > self._ttl
            ]
            for sid in expired:
                self._data.pop(sid, None)
        return len(expired)


async def run_sweeper(store: InMemorySessionStore) -> None:
    """Background task: periodically evict expired sessions."""
    interval = get_settings().session_sweep_seconds
    while True:
        await asyncio.sleep(interval)
        await store.sweep()

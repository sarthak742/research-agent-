import pytest

from agent.models import Session, SessionStatus
from agent.store import InMemorySessionStore


def _session(sid="s1"):
    return Session(id=sid, topic="t", depth="basic")


async def test_create_and_get_roundtrip():
    store = InMemorySessionStore(ttl=100)
    s = _session()
    await store.create(s)
    got = await store.get("s1")
    assert got is s


async def test_get_missing_returns_none():
    store = InMemorySessionStore(ttl=100)
    assert await store.get("nope") is None


async def test_expired_session_is_evicted_on_get():
    store = InMemorySessionStore(ttl=-1)  # everything is already expired
    await store.create(_session())
    assert await store.get("s1") is None


async def test_save_updates_session():
    store = InMemorySessionStore(ttl=100)
    s = _session()
    await store.create(s)
    s.status = SessionStatus.COMPLETED
    s.report = "done"
    await store.save(s)
    got = await store.get("s1")
    assert got.status == SessionStatus.COMPLETED
    assert got.report == "done"


async def test_sweep_removes_expired():
    store = InMemorySessionStore(ttl=-1)
    await store.create(_session("a"))
    await store.create(_session("b"))
    removed = await store.sweep()
    assert removed == 2

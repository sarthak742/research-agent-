import pytest

from agent.search import SearchClient, SearchError


def _ok_result(**_kw):
    return {"results": [{"title": "T", "url": "http://x", "content": "c"}]}


async def test_search_returns_results():
    client = SearchClient(search_fn=_ok_result)
    results = await client.search("hello")
    assert results == [{"title": "T", "url": "http://x", "content": "c"}]


async def test_search_caches_identical_query():
    calls = {"n": 0}

    def counting(**_kw):
        calls["n"] += 1
        return _ok_result()

    client = SearchClient(search_fn=counting)
    await client.search("same")
    await client.search("same")
    assert calls["n"] == 1  # second call served from cache


async def test_search_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return _ok_result()

    client = SearchClient(search_fn=flaky)
    client._base_delay = 0  # no backoff wait in tests
    results = await client.search("q")
    assert calls["n"] == 2
    assert len(results) == 1


async def test_search_raises_after_max_retries():
    def always_fail(**_kw):
        raise ConnectionError("down")

    client = SearchClient(search_fn=always_fail)
    client._base_delay = 0
    with pytest.raises(SearchError):
        await client.search("q")


async def test_search_handles_missing_results_key():
    client = SearchClient(search_fn=lambda **_kw: {})
    assert await client.search("q") == []

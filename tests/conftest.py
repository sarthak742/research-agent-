"""Shared test fakes.

None of the tests touch the network. We inject fakes for the LLM and the
search client, which is exactly what the dependency-injection seams in
llm.py / search.py / core.py were built for.
"""
import sys
from pathlib import Path

# Make the project root importable when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agent.search import SearchError


class FakeLLM:
    """Stand-in for LLMClient. Returns queued responses for ``complete`` and
    streams ``stream_text`` word-by-word for ``stream``."""

    def __init__(self, responses=None, stream_text="Report body [1]."):
        self._responses = list(responses or [])
        self._stream_text = stream_text
        self.complete_calls = []

    async def complete(self, prompt, temperature):
        self.complete_calls.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return "[]"

    async def stream(self, prompt, temperature):
        for word in self._stream_text.split(" "):
            yield word + " "


class FakeSearch:
    """Stand-in for SearchClient with per-query canned results."""

    def __init__(self, mapping=None, fail=None):
        self.mapping = mapping or {}
        self.fail = set(fail or [])
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        if query in self.fail:
            raise SearchError(query)
        return self.mapping.get(
            query,
            [{"title": f"T-{query}", "url": f"http://x/{query}", "content": "content"}],
        )


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_search():
    return FakeSearch()

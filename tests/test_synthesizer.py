import pytest

from agent.models import Source
from agent.synthesizer import Synthesizer, build_sources_block
from tests.conftest import FakeLLM


def _sources(n):
    return [
        Source(id=i + 1, title=f"Title {i+1}", url=f"http://s/{i+1}", content="body")
        for i in range(n)
    ]


def test_build_sources_block_numbered():
    block = build_sources_block(_sources(2))
    assert "[1] Title 1 — http://s/1" in block
    assert "[2] Title 2 — http://s/2" in block


async def test_synthesize_appends_sources_section():
    llm = FakeLLM(stream_text="")  # not used by non-stream path
    synth = Synthesizer(llm)
    # override complete to return a fixed body
    async def fake_complete(prompt, temperature):
        return "## Executive Summary\nFinding [1]."
    llm.complete = fake_complete
    report = await synth.synthesize("topic", _sources(1))
    assert "## Sources" in report
    assert "[1] Title 1" in report
    assert "Finding [1]." in report


async def test_synthesize_stream_yields_chunks():
    llm = FakeLLM(stream_text="hello world foo")
    synth = Synthesizer(llm)
    chunks = [c async for c in synth.synthesize_stream("topic", _sources(1))]
    assert "".join(chunks).strip() == "hello world foo"


async def test_synthesizer_caps_sources(monkeypatch):
    llm = FakeLLM()
    synth = Synthesizer(llm)
    # 20 sources but the cap (default 15) should limit what enters the prompt.
    capped = synth._cap(_sources(20))
    assert len(capped) == 15

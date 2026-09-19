import pytest

from agent.core import ResearchAgent
from agent.models import EventType, Session, SessionStatus
from agent.planner import Planner
from agent.synthesizer import Synthesizer
from tests.conftest import FakeLLM, FakeSearch


def _agent(llm, search):
    return ResearchAgent(llm, search, Planner(llm), Synthesizer(llm))


async def _collect(agent, session):
    return [e async for e in agent.run(session)]


async def test_happy_path_produces_report_and_completes():
    llm = FakeLLM(responses=['["q1", "q2"]', "[]"], stream_text="Summary [1].")
    search = FakeSearch()
    agent = _agent(llm, search)
    session = Session(id="s", topic="topic", depth="basic")

    events = await _collect(agent, session)
    types = [e.type for e in events]

    assert EventType.PLANNING in types
    assert EventType.READING in types
    assert EventType.WRITING in types
    assert EventType.TOKEN in types
    assert events[-1].type == EventType.DONE
    assert session.status == SessionStatus.COMPLETED
    assert "## Sources" in session.report
    assert "Summary [1]." in session.report


async def test_sources_are_deduped_by_url():
    llm = FakeLLM(responses=['["q1", "q2"]', "[]"])
    # both queries return the SAME url -> should collapse to one source.
    dup = [{"title": "T", "url": "http://dup", "content": "c"}]
    search = FakeSearch(mapping={"q1": dup, "q2": dup})
    agent = _agent(llm, search)
    session = Session(id="s", topic="t", depth="basic")

    await _collect(agent, session)
    assert len(session.sources) == 1
    assert session.sources[0].id == 1


async def test_global_search_budget_is_respected():
    # 6 initial (detailed ceiling) + reflect asks for 6 more, but the global
    # cap (max_total_searches=10) must clamp the second round to 4.
    initial = "[" + ",".join(f'"q{i}"' for i in range(6)) + "]"
    followups = "[" + ",".join(f'"r{i}"' for i in range(6)) + "]"
    llm = FakeLLM(responses=[initial, followups, "[]"])
    search = FakeSearch()
    agent = _agent(llm, search)
    session = Session(id="s", topic="t", depth="detailed")

    await _collect(agent, session)
    assert len(search.queries) == 10  # never exceeds max_total_searches


async def test_search_failure_is_reported_but_run_continues():
    llm = FakeLLM(responses=['["q1", "q2"]', "[]"], stream_text="body")
    search = FakeSearch(fail={"q1"})
    agent = _agent(llm, search)
    session = Session(id="s", topic="t", depth="basic")

    events = await _collect(agent, session)
    types = [e.type for e in events]
    assert EventType.ERROR in types            # q1 failure surfaced
    assert events[-1].type == EventType.DONE   # run still finished
    assert session.status == SessionStatus.COMPLETED
    assert len(session.sources) == 1           # q2 still contributed

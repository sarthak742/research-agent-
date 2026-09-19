import pytest

from agent.planner import Planner, _extract_json_array
from tests.conftest import FakeLLM


# --- JSON extraction robustness ---
def test_extract_plain_array():
    assert _extract_json_array('["a", "b"]') == ["a", "b"]


def test_extract_from_code_fence_and_prose():
    text = 'Here is the plan:\n```json\n["one", "two"]\n```\nHope that helps!'
    assert _extract_json_array(text) == ["one", "two"]


def test_extract_bad_json_returns_empty():
    assert _extract_json_array("not json at all") == []
    assert _extract_json_array('["unterminated') == []


def test_extract_filters_blanks():
    assert _extract_json_array('["a", "", "  ", "b"]') == ["a", "b"]


# --- initial_plan ---
async def test_initial_plan_parses_and_returns_queries():
    llm = FakeLLM(responses=['["q1", "q2", "q3"]'])
    planner = Planner(llm)
    queries = await planner.initial_plan("some topic", "detailed")
    assert queries == ["q1", "q2", "q3"]


async def test_initial_plan_falls_back_to_topic_on_garbage():
    llm = FakeLLM(responses=["the model rambled with no array"])
    planner = Planner(llm)
    queries = await planner.initial_plan("quantum computing", "basic")
    assert queries == ["quantum computing"]


async def test_initial_plan_respects_ceiling():
    # detailed ceiling is max_queries (6 by default); ask for more.
    many = "[" + ",".join(f'"q{i}"' for i in range(20)) + "]"
    llm = FakeLLM(responses=[many])
    planner = Planner(llm)
    queries = await planner.initial_plan("broad topic", "detailed")
    assert len(queries) <= 6


# --- reflect ---
async def test_reflect_returns_empty_when_model_says_enough():
    llm = FakeLLM(responses=["[]"])
    planner = Planner(llm)
    followups = await planner.reflect("topic", sources=[], searches_used=2)
    assert followups == []


async def test_reflect_returns_followups():
    llm = FakeLLM(responses=['["gap query"]'])
    planner = Planner(llm)
    followups = await planner.reflect("topic", sources=[], searches_used=2)
    assert followups == ["gap query"]


async def test_reflect_stops_when_budget_exhausted():
    # searches_used at/over max_total_searches -> no further searching, and
    # the LLM should not even be consulted.
    llm = FakeLLM(responses=['["should not be used"]'])
    planner = Planner(llm)
    followups = await planner.reflect("topic", sources=[], searches_used=999)
    assert followups == []
    assert llm.complete_calls == []  # short-circuited before calling the LLM

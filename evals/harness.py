"""Shared helpers for the eval scripts: run a real research session and
save scorecards. These require API keys (they hit Groq + Tavily)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agent.core import build_agent
from agent.llm import LLMClient
from agent.models import Session

RESULTS_DIR = Path(__file__).resolve().parent / "results"


async def run_research(topic: str, depth: str = "basic") -> Session:
    """Drive one full research run and return the finished session."""
    agent = build_agent()
    session = Session(id=f"eval-{topic[:20]}", topic=topic, depth=depth)
    async for _event in agent.run(session):
        pass  # we only need the final session state for evals
    return session


def judge_llm() -> LLMClient:
    """A separate LLM client used only for judging (temperature 0 in judges)."""
    return LLMClient()


def load_topics() -> list[dict]:
    path = Path(__file__).resolve().parent / "datasets" / "topics.json"
    return json.loads(path.read_text())


def save_scorecard(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{name}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def have_keys() -> bool:
    return bool(os.getenv("GROQ_API_KEY") and os.getenv("TAVILY_API_KEY"))

"""Planning + reflection — the part that makes this an *agent*, not a pipeline.

The original version planned once and ran a fixed number of searches
(``3 if basic else 7``), hardcoded and chosen by a UI button. That is a
pipeline. Here the LLM actually makes two kinds of decisions:

* ``initial_plan`` — given a topic, decide *which* queries to run AND *how
  many* (within safety bounds). A broad topic gets more queries than a
  narrow one. This is the "decides based on complexity" behaviour.

* ``reflect`` — after seeing what came back, decide whether there is a gap
  worth another round of searching, and if so what to search for. Returning
  an empty list means "I have enough, stop." This closes the loop.

Both return plain Python lists so ``core.py`` stays simple and testable.
"""
from __future__ import annotations

import json
import logging
import re

from agent.llm import LLMClient
from agent.models import Source
from config import get_settings

logger = logging.getLogger(__name__)


def _extract_json_array(text: str) -> list[str]:
    """Pull a JSON array of strings out of an LLM response.

    LLMs wrap JSON in prose or code fences, so we can't just json.loads the
    whole thing. We find the first [...] block and parse that, and we fall
    back to an empty list rather than raising — an unparseable reflection
    just means "no follow-ups", which is a safe default.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("could not parse JSON array from LLM output")
        return []
    if not isinstance(parsed, list):
        return []
    # Keep only non-empty strings, trimmed.
    return [str(x).strip() for x in parsed if str(x).strip()]


class Planner:
    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._s = get_settings()

    async def initial_plan(self, topic: str, depth: str) -> list[str]:
        """Decide the opening set of search queries (count included)."""
        # ``depth`` nudges the ceiling but the model still chooses the number.
        ceiling = self._s.max_queries if depth == "detailed" else max(
            self._s.min_queries + 1, self._s.max_queries // 2
        )
        prompt = f"""You are planning web research on the topic: "{topic}".

Decide how many distinct search queries are needed to cover this topic well.
A narrow, specific topic needs fewer; a broad or multi-faceted one needs more.
Choose between {self._s.min_queries} and {ceiling} queries based on the
topic's breadth — do not pad to the maximum.

Return ONLY a JSON array of search query strings, most important first.
Example: ["query one", "query two"]"""

        raw = await self._llm.complete(prompt, self._s.plan_temperature)
        queries = _extract_json_array(raw)

        # Guardrails: never zero, never over the ceiling.
        if not queries:
            queries = [topic]
        return queries[:ceiling]

    async def reflect(
        self, topic: str, sources: list[Source], searches_used: int
    ) -> list[str]:
        """Look at what we've gathered and decide on follow-up queries.

        Returns [] when the agent judges coverage sufficient, which ends the
        loop. Bounded so it can never blow past the global search budget.
        """
        remaining = self._s.max_total_searches - searches_used
        if remaining <= 0:
            return []

        digest = "\n".join(
            f"- {s.title}: {s.content[:160]}" for s in sources[:12]
        ) or "(no results yet)"

        prompt = f"""Research topic: "{topic}"

Here is a digest of what we have found so far:
{digest}

Identify important gaps or angles NOT yet covered. If the topic is already
well covered, return an empty array. Otherwise return up to {remaining} NEW
search queries that would fill the gaps (do not repeat earlier queries).

Return ONLY a JSON array of strings (possibly empty). Example: [] or ["gap query"]"""

        raw = await self._llm.complete(prompt, self._s.plan_temperature)
        followups = _extract_json_array(raw)
        return followups[:remaining]

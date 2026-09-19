"""The agent loop.

This is the orchestrator that turns the pieces into a real
plan -> act -> reflect -> act -> synthesize loop, emitting an ``Event`` for
every step so the UI can show progress live.

The loop is *bounded* on purpose (``max_rounds``, ``max_total_searches``):
an autonomous loop that can decide to keep searching also needs a leash, or
one topic could rack up unbounded cost and latency. Being able to explain
those bounds is part of defending the design.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from agent.llm import LLMClient
from agent.models import Event, EventType, Session, SessionStatus, Source
from agent.planner import Planner
from agent.search import SearchClient, SearchError
from agent.synthesizer import Synthesizer
from config import get_settings

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(
        self,
        llm: LLMClient,
        search: SearchClient,
        planner: Planner,
        synthesizer: Synthesizer,
    ):
        self._llm = llm
        self._search = search
        self._planner = planner
        self._synth = synthesizer
        self._s = get_settings()

    async def run(self, session: Session) -> AsyncIterator[Event]:
        """Drive one research session, yielding events as work happens.

        Mutates ``session`` in place with the final report/sources/status so
        the caller (API layer) can persist it.
        """
        try:
            session.status = SessionStatus.RUNNING
            async for event in self._run_inner(session):
                yield event
        except Exception as exc:  # last-resort guard so the stream always ends
            logger.exception("research run failed")
            session.status = SessionStatus.FAILED
            session.error = str(exc)
            yield Event(EventType.ERROR, f"Research failed: {exc}")

    async def _run_inner(self, session: Session) -> AsyncIterator[Event]:
        topic = session.topic
        yield Event(EventType.THINKING, "Analyzing the topic...")

        # url -> Source, so we can dedupe and keep citation ids stable.
        seen: dict[str, Source] = {}
        searches_used = 0

        # --- Round 1: initial plan ---
        yield Event(EventType.PLANNING, "Planning research questions...")
        queries = await self._planner.initial_plan(topic, session.depth)
        yield Event(
            EventType.PLANNING,
            f"Planned {len(queries)} search{'es' if len(queries) != 1 else ''}.",
            {"queries": queries},
        )

        for round_no in range(1, self._s.max_rounds + 1):
            if not queries:
                break

            # Respect the global budget even if the planner over-asked.
            budget_left = self._s.max_total_searches - searches_used
            queries = queries[:budget_left]
            if not queries:
                break

            # Run this round's searches concurrently.
            async for ev in self._run_searches(queries, seen):
                yield ev
            searches_used += len(queries)
            session.sources = list(seen.values())

            # Decide whether to go another round.
            if round_no >= self._s.max_rounds or searches_used >= self._s.max_total_searches:
                break
            yield Event(EventType.REFLECTING, "Checking for gaps in the research...")
            queries = await self._planner.reflect(
                topic, session.sources, searches_used
            )
            if queries:
                yield Event(
                    EventType.REFLECTING,
                    f"Found gaps — running {len(queries)} more search(es).",
                    {"queries": queries},
                )
            else:
                yield Event(EventType.REFLECTING, "Coverage looks sufficient.")

        # --- Synthesis (streamed) ---
        yield Event(EventType.WRITING, "Synthesizing report...")
        body_parts: list[str] = []
        async for chunk in self._synth.synthesize_stream(topic, session.sources):
            body_parts.append(chunk)
            yield Event(EventType.TOKEN, chunk)

        # Assemble the final report (body + numbered sources block).
        from agent.synthesizer import build_sources_block
        capped = session.sources[: self._s.max_sources_in_report]
        report = "".join(body_parts).strip()
        report = f"{report}\n\n## Sources\n{build_sources_block(capped)}\n"
        session.report = report
        session.status = SessionStatus.COMPLETED

        yield Event(EventType.DONE, "Research complete.", {"report": report})

    async def _run_searches(
        self, queries: list[str], seen: dict[str, Source]
    ) -> AsyncIterator[Event]:
        """Search a batch of queries concurrently and fold results into ``seen``."""
        results = await asyncio.gather(
            *(self._safe_search(q) for q in queries), return_exceptions=False
        )
        for q, res in zip(queries, results):
            if res is None:  # a query that failed even after retries
                yield Event(EventType.ERROR, f"Search failed: {q}")
                continue
            added = 0
            for r in res:
                url = r.get("url", "")
                if not url or url in seen:
                    continue
                src = Source(
                    id=len(seen) + 1,
                    title=r.get("title", "Untitled"),
                    url=url,
                    content=r.get("content", ""),
                )
                seen[url] = src
                added += 1
            yield Event(
                EventType.READING,
                f'Read {added} new source(s) for: "{q}"',
            )

    async def _safe_search(self, query: str) -> list[dict] | None:
        try:
            return await self._search.search(query)
        except SearchError:
            return None


def build_agent() -> ResearchAgent:
    """Wire up a real agent from settings (used by the API layer)."""
    llm = LLMClient()
    search = SearchClient()
    return ResearchAgent(llm, search, Planner(llm), Synthesizer(llm))

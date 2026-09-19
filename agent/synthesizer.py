"""Report synthesis with inline, checkable citations.

The original synthesizer just appended a flat list of source titles at the
bottom. There was no link between any sentence in the report and the source
it came from, so a reader couldn't verify a single claim — and the LLM was
free to invent things with no traceability.

Here every source gets a stable number, the prompt instructs the model to
cite claims inline as [n], and ``build_source_list`` renders a numbered
reference section those [n] markers point at. That is what makes the output
*checkable* — the whole reason the project exists.

``synthesize_stream`` yields the report token-by-token so the UI can render
it live instead of waiting for the whole thing.
"""
from __future__ import annotations

from typing import AsyncIterator

from agent.llm import LLMClient
from agent.models import Source
from config import get_settings


def build_sources_block(sources: list[Source]) -> str:
    """Render the numbered reference list that [n] citations point to."""
    lines = [f"[{s.id}] {s.title} — {s.url}" for s in sources]
    return "\n".join(lines)


def _build_prompt(topic: str, sources: list[Source]) -> str:
    numbered = "\n".join(
        f"[{s.id}] {s.title}\n{s.content[:500]}" for s in sources
    ) or "(no sources found)"

    return f"""Write an executive research report on: "{topic}"

Use ONLY the numbered sources below. When you state a fact, cite the source
it came from inline using its number in square brackets, e.g. "Adoption grew
sharply in 2024 [2]." Every substantive claim must carry at least one
citation. Do NOT invent sources or cite numbers that are not listed. If the
sources do not cover something, say so rather than guessing.

SOURCES:
{numbered}

Produce this exact structure in Markdown:

## Executive Summary
A short overview (2-4 sentences), with citations.

## Key Findings
The main points as a bulleted list, each finding citing its source(s).

## Limitations
One or two sentences on what the available sources did NOT cover.

Do not include a Sources section; it will be appended automatically."""


class Synthesizer:
    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._s = get_settings()

    def _cap(self, sources: list[Source]) -> list[Source]:
        return sources[: self._s.max_sources_in_report]

    async def synthesize(self, topic: str, sources: list[Source]) -> str:
        sources = self._cap(sources)
        prompt = _build_prompt(topic, sources)
        body = await self._llm.complete(prompt, self._s.report_temperature)
        return self._finalize(body, sources)

    async def synthesize_stream(
        self, topic: str, sources: list[Source]
    ) -> AsyncIterator[str]:
        """Yield report chunks as they generate; caller appends the sources."""
        sources = self._cap(sources)
        prompt = _build_prompt(topic, sources)
        async for chunk in self._llm.stream(prompt, self._s.report_temperature):
            yield chunk

    def _finalize(self, body: str, sources: list[Source]) -> str:
        return f"{body.strip()}\n\n## Sources\n{build_sources_block(sources)}\n"

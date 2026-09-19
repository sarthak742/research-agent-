"""LLM-as-judge helpers.

These wrap the LLM to produce numeric/boolean judgements used by the eval
scripts. Each parser is separated from the LLM call so the parsing can be
unit-tested without the network.

Note the obvious limitation, which you should state out loud in an interview:
using an LLM to judge an LLM's output is convenient but not ground truth — the
judge shares some of the same blind spots. It's a signal, not a certificate.
For a stronger eval you'd add a small human-labelled set and check the judge
against it.
"""
from __future__ import annotations

import re


def parse_score(text: str, lo: int = 0, hi: int = 5) -> float:
    """Pull the first integer in [lo, hi] out of a judge response and
    normalise it to 0..1. Returns 0.0 if nothing parseable is found."""
    for match in re.findall(r"-?\d+", text):
        val = int(match)
        if lo <= val <= hi:
            return (val - lo) / (hi - lo)
    return 0.0


def parse_yes_no(text: str) -> bool:
    """True if the response starts with / clearly says yes."""
    t = text.strip().lower()
    return t.startswith("yes") or bool(re.match(r"^(true|supported|1)\b", t))


async def judge_relevance(llm, topic: str, query: str) -> float:
    prompt = f"""On a scale of 0 to 5, how relevant and useful is this search
query for researching the topic?

Topic: {topic}
Query: {query}

Answer with ONLY a single integer 0-5."""
    out = await llm.complete(prompt, temperature=0.0)
    return parse_score(out)


async def judge_support(llm, claim: str, source_text: str) -> bool:
    prompt = f"""Does the SOURCE support the CLAIM? Answer only "yes" or "no".

CLAIM: {claim}

SOURCE: {source_text[:800]}"""
    out = await llm.complete(prompt, temperature=0.0)
    return parse_yes_no(out)


async def judge_report(llm, topic: str, report: str) -> float:
    prompt = f"""Rate this research report on the topic "{topic}" from 0 to 5,
considering: coverage of the topic, clear structure (Executive Summary, Key
Findings, Limitations), and whether claims carry citations.

REPORT:
{report[:3000]}

Answer with ONLY a single integer 0-5."""
    out = await llm.complete(prompt, temperature=0.0)
    return parse_score(out)

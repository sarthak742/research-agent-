"""Eval: planning quality.

For each topic, generate the plan and score it on:
* relevance   — LLM judge rates each query's usefulness (0..1), averaged
* diversity   — fraction of query pairs that are not near-duplicates
                (lexical Jaccard < 0.6), so the plan isn't 5 rephrasings
* count       — how many queries the planner chose (sanity vs. bounds)

Live only (needs API keys):
    python -m evals.eval_planning
"""
from __future__ import annotations

import asyncio
import json
from itertools import combinations

from agent.llm import LLMClient
from agent.planner import Planner
from evals.harness import have_keys, load_topics, save_scorecard
from evals.judges import judge_relevance


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def diversity(queries: list[str]) -> float:
    pairs = list(combinations(queries, 2))
    if not pairs:
        return 1.0
    distinct = sum(1 for a, b in pairs if _jaccard(a, b) < 0.6)
    return round(distinct / len(pairs), 3)


async def run_live() -> dict:
    llm = LLMClient()
    planner = Planner(llm)
    results = []
    for item in load_topics():
        topic, depth = item["topic"], item.get("depth", "basic")
        queries = await planner.initial_plan(topic, depth)
        rel = [await judge_relevance(llm, topic, q) for q in queries]
        results.append({
            "topic": topic,
            "query_count": len(queries),
            "avg_relevance": round(sum(rel) / len(rel), 3) if rel else 0.0,
            "diversity": diversity(queries),
            "queries": queries,
        })
    return {"eval": "planning", "results": results}


def main():
    if not have_keys():
        print("This eval needs GROQ_API_KEY + TAVILY_API_KEY. Aborting.")
        return
    payload = asyncio.run(run_live())
    print(json.dumps(payload, indent=2))
    print(f"\nScorecard saved to {save_scorecard('planning', payload)}")


if __name__ == "__main__":
    main()

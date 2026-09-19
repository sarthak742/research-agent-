"""Eval: citation grounding.

For each topic, run a full research session, then for every inline [n]
citation ask an LLM judge whether source n actually supports the sentence.
Reports: % of claims that are cited, % of citations that are grounded, count
of invalid (fabricated) citation numbers, and the list of uncited claims.

Run offline against the recorded fixture:
    python -m evals.eval_grounding --fixture

Run live (needs GROQ_API_KEY + TAVILY_API_KEY):
    python -m evals.eval_grounding
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evals.grounding import extract_claims, score_grounding
from evals.harness import have_keys, judge_llm, load_topics, run_research, save_scorecard
from evals.judges import judge_support


async def _verdicts_for(llm, claims, source_by_id):
    """Ask the judge about every (claim, cited source) pair."""
    verdicts: dict[tuple[int, int], bool] = {}
    for c in claims:
        for sid in c.citations:
            src = source_by_id.get(sid)
            if src is None:
                continue
            verdicts[(c.index, sid)] = await judge_support(llm, c.text, src["content"])
    return verdicts


async def evaluate_report(report: str, sources: list[dict], llm) -> dict:
    claims = extract_claims(report)
    source_by_id = {s["id"]: s for s in sources}
    verdicts = await _verdicts_for(llm, claims, source_by_id)
    return score_grounding(claims, set(source_by_id), verdicts)


async def run_live() -> dict:
    llm = judge_llm()
    results = []
    for item in load_topics():
        session = await run_research(item["topic"], item.get("depth", "basic"))
        sources = [
            {"id": s.id, "title": s.title, "url": s.url, "content": s.content}
            for s in session.sources
        ]
        metrics = await evaluate_report(session.report, sources, llm)
        results.append({"topic": item["topic"], **metrics})
    return {"eval": "grounding", "mode": "live", "results": results}


def run_fixture() -> dict:
    """Deterministic offline run using recorded data + a stub judge that
    marks a citation grounded iff the claim's keyword appears in the source.
    Demonstrates the pipeline without API keys."""
    fx = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "sample_run.json").read_text()
    )
    claims = extract_claims(fx["report"])
    source_by_id = {s["id"]: s for s in fx["sources"]}

    verdicts: dict[tuple[int, int], bool] = {}
    for c in claims:
        for sid in c.citations:
            src = source_by_id.get(sid)
            if src is None:
                continue
            # Stub judge: grounded if any 5+ char word of the claim is in source.
            words = {w.lower().strip(".,") for w in c.text.split() if len(w) >= 5}
            verdicts[(c.index, sid)] = any(w in src["content"].lower() for w in words)

    metrics = score_grounding(claims, set(source_by_id), verdicts)
    return {"eval": "grounding", "mode": "fixture", "results": [
        {"topic": fx["topic"], **metrics}
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="run offline on the fixture")
    args = ap.parse_args()

    if args.fixture or not have_keys():
        if not args.fixture:
            print("No API keys found — running in fixture mode.\n")
        payload = run_fixture()
    else:
        payload = asyncio.run(run_live())

    print(json.dumps(payload, indent=2))
    path = save_scorecard("grounding", payload)
    print(f"\nScorecard saved to {path}")


if __name__ == "__main__":
    main()

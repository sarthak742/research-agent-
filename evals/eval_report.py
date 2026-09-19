"""Eval: overall report quality (LLM-as-judge rubric).

For each topic, run the research and have a judge rate the report 0..1 on
coverage, structure, and citation presence. Coarser than the grounding eval,
but a useful single number to track across changes.

Live only (needs API keys):
    python -m evals.eval_report
"""
from __future__ import annotations

import asyncio
import json

from evals.harness import have_keys, judge_llm, load_topics, run_research, save_scorecard
from evals.judges import judge_report


async def run_live() -> dict:
    llm = judge_llm()
    results = []
    for item in load_topics():
        session = await run_research(item["topic"], item.get("depth", "basic"))
        score = await judge_report(llm, item["topic"], session.report)
        results.append({
            "topic": item["topic"],
            "quality_score": score,
            "source_count": len(session.sources),
            "report_chars": len(session.report),
        })
    avg = round(sum(r["quality_score"] for r in results) / len(results), 3) if results else 0
    return {"eval": "report_quality", "average_score": avg, "results": results}


def main():
    if not have_keys():
        print("This eval needs GROQ_API_KEY + TAVILY_API_KEY. Aborting.")
        return
    payload = asyncio.run(run_live())
    print(json.dumps(payload, indent=2))
    print(f"\nScorecard saved to {save_scorecard('report_quality', payload)}")


if __name__ == "__main__":
    main()

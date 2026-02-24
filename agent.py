import asyncio
import json
import uuid
import os
from typing import AsyncGenerator, List, Dict, Any
from groq import Groq
from tavily import Client as TavilyClient

class ResearchAgent:
    def __init__(self, groq_key: str, tavily_key: str):
        self.groq = Groq(api_key=groq_key)
        self.tavily = TavilyClient(api_key=tavily_key)
        self.sessions: Dict[str, dict] = {}

    async def research(self, topic: str, depth: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "topic": topic,
            "depth": depth,
            "status": "running",
            "plan": [],
            "searches": [],
            "report": ""
        }
        return session_id

    async def stream(self, session_id: str) -> AsyncGenerator[dict, None]:
        session = self.sessions.get(session_id)
        if not session:
            yield {"type": "error", "message": "Session not found"}
            return

        # Yield thinking
        yield {"type": "thinking", "message": "Analyzing topic and planning research approach..."}

        # Create plan
        plan = await self._create_plan(session["topic"], session["depth"])
        session["plan"] = plan

        # Dynamic search count
        search_count = 3 if session["depth"] == "basic" else 7

        # Perform searches
        for i, aspect in enumerate(plan[:search_count]):
            yield {"type": "searching", "message": f"Searching for: {aspect}"}

            try:
                results = self.tavily.search(query=aspect, max_results=3)
                session["searches"].extend(results.get("results", []))
            except Exception as e:
                yield {"type": "error", "message": f"Search failed: {str(e)}"}
                continue

            yield {"type": "reading", "message": f"Reading {len(results.get('results', []))} results"}

            await asyncio.sleep(0.5)

        # Synthesize
        yield {"type": "writing", "message": "Synthesizing research report..."}
        report = await self._synthesize_report(session["topic"], session["searches"], session["depth"])
        session["report"] = report
        session["status"] = "completed"

        yield {"type": "done", "message": "Research complete!", "report": report}

    async def _create_plan(self, topic: str, depth: str) -> List[str]:
        prompt = f"""Create a brief research plan for: {topic}
        List {3 if depth == 'basic' else 7} specific aspects to research.
        Return only a JSON array of strings, nothing else."""

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content
        # Parse JSON array from response
        import re
        match = re.search(r'\[.*\]', content)
        if match:
            import json
            return json.loads(match.group())
        return [topic]

    async def _synthesize_report(self, topic: str, searches: List[dict], depth: str) -> str:
        sources = "\n".join([f"- {s.get('title', 'Unknown')} ({s.get('url', '')})" for s in searches[:10]])

        prompt = f"""Write an executive-style research report on: {topic}

Sources:
{sources}

Format:
## Executive Summary
[Brief overview]

## Key Findings
[Main points from research]

## Sources
{sources}
"""

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content

# Global agent instance
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        groq_key = os.getenv("GROQ_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not groq_key or not tavily_key:
            raise ValueError("API keys not configured")
        _agent = ResearchAgent(groq_key, tavily_key)
    return _agent

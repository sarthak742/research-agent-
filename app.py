"""FastAPI application: routes, real SSE streaming, wiring.

Fixes over the original app.py:

* Real Server-Sent Events via ``sse_starlette.EventSourceResponse`` and a
  proper ``EventSource`` client, instead of the broken
  ``sse_starlette.SSEEvent`` call (that symbol doesn't exist) plus a manual
  fetch-reader on the frontend.
* Correct HTTP status codes (the original ``return {...}, 404`` returned a
  tuple FastAPI serialized as a 200 body, so "not found" looked like success).
* Input validation via the pydantic ``ResearchRequest`` model.
* CORS locked to configured origins instead of ``allow_origins=["*"]``.
* Session store with TTL cleanup, plus a background sweeper started in the
  app lifespan.
* Structured logging.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from agent.core import build_agent
from agent.models import ResearchRequest, Session, SessionStatus
from agent.store import InMemorySessionStore, run_sweeper
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("research_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.store = InMemorySessionStore()
    # The agent is built lazily on first use so the app can boot (and serve
    # the health check / frontend) even before API keys are configured.
    app.state.agent = None
    sweeper = asyncio.create_task(run_sweeper(app.state.store))
    logger.info("research agent started")
    try:
        yield
    finally:
        sweeper.cancel()
        logger.info("research agent stopped")


app = FastAPI(title="Research Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_agent(app: FastAPI):
    if app.state.agent is None:
        settings = app.state.settings
        if not settings.groq_api_key or not settings.tavily_api_key:
            raise HTTPException(
                status_code=503,
                detail="Server missing GROQ_API_KEY / TAVILY_API_KEY.",
            )
        app.state.agent = build_agent()
    return app.state.agent


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/research")
async def create_research(req: ResearchRequest, request: Request):
    # Validating the agent is available before creating a session avoids
    # orphan sessions that can never run.
    _get_agent(request.app)
    session = Session(id=_new_id(), topic=req.topic, depth=req.depth)
    await request.app.state.store.create(session)
    logger.info("created session %s for topic %r", session.id, req.topic)
    return {"research_id": session.id}


@app.get("/stream/{research_id}")
async def stream_research(research_id: str, request: Request):
    store = request.app.state.store
    session = await store.get(research_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    agent = _get_agent(request.app)

    async def event_generator():
        try:
            async for event in agent.run(session):
                # If the client disconnected, stop doing (expensive) work.
                if await request.is_disconnected():
                    logger.info("client disconnected from %s", research_id)
                    break
                yield {"data": json.dumps(event.to_dict())}
        finally:
            await store.save(session)

    return EventSourceResponse(event_generator())


@app.get("/report/{research_id}")
async def get_report(research_id: str, request: Request):
    session = await request.app.state.store.get(research_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(
        {
            "status": session.status.value,
            "report": session.report,
            "sources": [s.to_dict() for s in session.sources],
            "error": session.error,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port)

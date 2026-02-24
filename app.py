from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str
    depth: str = "basic"

# Import agent after environment is loaded
from agent import get_agent

@app.post("/research")
async def create_research(req: ResearchRequest):
    agent = get_agent()
    research_id = await agent.research(req.topic, req.depth)
    return {"research_id": research_id}

@app.get("/stream/{research_id}")
async def stream_research(request: Request, research_id: str):
    import sse_starlette as sse

    async def event_generator():
        agent = get_agent()
        async for event in agent.stream(research_id):
            yield sse.SSEEvent(data=json.dumps(event))

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/report/{research_id}")
async def get_report(research_id: str):
    agent = get_agent()
    session = agent.sessions.get(research_id)
    if not session:
        return {"error": "Session not found"}, 404
    return {"report": session.get("report", "")}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

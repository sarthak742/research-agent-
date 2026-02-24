from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
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

@app.post("/research")
async def create_research(req: ResearchRequest):
    return {"research_id": "test-123"}

@app.get("/stream/{research_id}")
async def stream_research(request: Request, research_id: str):
    async def event_generator():
        yield {"event": "thinking", "data": "Starting research..."}
        yield {"event": "done", "data": "Research complete"}
    return event_generator()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

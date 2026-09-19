import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from agent.models import Event, EventType, Session, SessionStatus


class FakeAgent:
    async def run(self, session: Session):
        session.status = SessionStatus.RUNNING
        yield Event(EventType.PLANNING, "planning")
        session.report = "REPORT [1]\n\n## Sources\n[1] T — http://x"
        session.status = SessionStatus.COMPLETED
        yield Event(EventType.DONE, "done", {"report": session.report})


@pytest.fixture
def client():
    with TestClient(app_module.app) as c:
        # Inject a fake agent so no API keys / network are needed.
        c.app.state.agent = FakeAgent()
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_research_validation_rejects_short_topic(client):
    r = client.post("/research", json={"topic": "ab", "depth": "basic"})
    assert r.status_code == 422  # pydantic min_length


def test_research_validation_rejects_bad_depth(client):
    r = client.post("/research", json={"topic": "valid topic", "depth": "huge"})
    assert r.status_code == 422


def test_create_research_returns_id_and_stores_session(client):
    r = client.post("/research", json={"topic": "quantum computing", "depth": "basic"})
    assert r.status_code == 200
    rid = r.json()["research_id"]
    assert rid
    # report endpoint should now find the (pending) session
    rep = client.get(f"/report/{rid}")
    assert rep.status_code == 200
    assert rep.json()["status"] == "pending"


def test_report_unknown_is_404(client):
    assert client.get("/report/does-not-exist").status_code == 404


def test_stream_unknown_is_404(client):
    assert client.get("/stream/does-not-exist").status_code == 404


def test_full_stream_flow(client):
    rid = client.post(
        "/research", json={"topic": "quantum computing", "depth": "basic"}
    ).json()["research_id"]

    r = client.get(f"/stream/{rid}")
    assert r.status_code == 200
    # SSE frames look like: "data: {...}\n\n"
    payloads = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    types = [p["type"] for p in payloads]
    assert "planning" in types
    assert "done" in types

    # After the stream, the stored report should be complete.
    rep = client.get(f"/report/{rid}").json()
    assert rep["status"] == "completed"
    assert "REPORT" in rep["report"]

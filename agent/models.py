"""Typed data structures shared across the agent.

Using dataclasses/pydantic here (instead of loose dicts like the original
version) means every event and record has a known shape, which is what
makes the code testable and the API contract stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    depth: str = "basic"

    @field_validator("topic")
    @classmethod
    def _strip_topic(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic must not be blank")
        return v

    @field_validator("depth")
    @classmethod
    def _check_depth(cls, v: str) -> str:
        if v not in ("basic", "detailed"):
            raise ValueError("depth must be 'basic' or 'detailed'")
        return v


# ---------------------------------------------------------------------------
# Domain records
# ---------------------------------------------------------------------------
@dataclass
class Source:
    """One web result, with a stable integer id used for [n] citations."""
    id: int
    title: str
    url: str
    content: str  # the snippet Tavily returns

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "url": self.url}


class EventType(str, Enum):
    THINKING = "thinking"
    PLANNING = "planning"
    SEARCHING = "searching"
    READING = "reading"
    REFLECTING = "reflecting"
    WRITING = "writing"
    TOKEN = "token"      # a chunk of the report as it streams
    DONE = "done"
    ERROR = "error"


@dataclass
class Event:
    """A single item in the live activity stream."""
    type: EventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type.value, "message": self.message}
        if self.data:
            out.update(self.data)
        return out


class SessionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Session:
    id: str
    topic: str
    depth: str
    status: SessionStatus = SessionStatus.PENDING
    report: str = ""
    sources: list[Source] = field(default_factory=list)
    error: str | None = None
    created_at: float = 0.0

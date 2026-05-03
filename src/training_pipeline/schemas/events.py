"""Canonical event and trajectory schemas.

Five event types model an LLM session:

- ``UserEvent``       — a user-authored message
- ``AssistantEvent``  — an assistant-authored message (text content)
- ``ToolCallEvent``   — the assistant requesting one or more tool invocations
- ``ToolResultEvent`` — the result of a tool invocation flowing back to the model
- ``ErrorEvent``      — a system or tool-level error attached to the trace

A ``Trajectory`` is an ordered sequence of events for one session. Trajectories
are the unit of redaction, tagging, validation, and export.

The schema is intentionally narrower than what production logs typically emit —
adapters in ``training_pipeline.ingest`` are responsible for mapping vendor
formats (OpenAI, Anthropic, generic chat) onto these types.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventRole(str, Enum):
    """Role identifier matching common chat-template conventions."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
    ERROR = "error"


class _BaseEvent(BaseModel):
    """Fields shared by every event type."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    event_id: str = Field(..., description="Stable id within a session")
    session_id: str = Field(..., description="Session this event belongs to")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


class UserEvent(_BaseEvent):
    kind: Literal["user"] = "user"
    role: Literal[EventRole.USER] = EventRole.USER
    content: str


class AssistantEvent(_BaseEvent):
    kind: Literal["assistant"] = "assistant"
    role: Literal[EventRole.ASSISTANT] = EventRole.ASSISTANT
    content: str = ""
    # When the assistant emits both text and tool calls, the content is the text
    # and the tool calls are stored on a ToolCallEvent that immediately follows.
    finish_reason: str | None = None


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Identifier matched by ToolResultEvent.tool_call_id")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallEvent(_BaseEvent):
    kind: Literal["tool_call"] = "tool_call"
    role: Literal[EventRole.ASSISTANT] = EventRole.ASSISTANT
    tool_calls: list[ToolCall]

    @field_validator("tool_calls")
    @classmethod
    def _at_least_one_call(cls, v: list[ToolCall]) -> list[ToolCall]:
        if not v:
            raise ValueError("ToolCallEvent must contain at least one ToolCall")
        return v


class ToolResultEvent(_BaseEvent):
    kind: Literal["tool_result"] = "tool_result"
    role: Literal[EventRole.TOOL] = EventRole.TOOL
    tool_call_id: str = Field(..., description="Matches ToolCall.id from a prior ToolCallEvent")
    name: str
    content: str = Field(..., description="Tool output, always serialised to a string")
    is_error: bool = False


class ErrorEvent(_BaseEvent):
    kind: Literal["error"] = "error"
    role: Literal[EventRole.ERROR] = EventRole.ERROR
    error_type: str
    message: str
    related_event_id: str | None = None


Event = Annotated[
    Union[UserEvent, AssistantEvent, ToolCallEvent, ToolResultEvent, ErrorEvent],
    Field(discriminator="kind"),
]
"""A discriminated union of every concrete event type."""


class Trajectory(BaseModel):
    """An ordered sequence of events for a single session.

    Trajectories are the primary unit of work for the pipeline. A trajectory is
    valid if its events are time-ordered and every ``ToolResultEvent`` references
    a prior ``ToolCallEvent`` from the same session.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    events: list[Event] = Field(default_factory=list)
    source: str | None = Field(
        default=None,
        description="Source label (e.g. 'production', 'staging', 'synthetic')",
    )
    domain: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

    @model_validator(mode="after")
    def _validate_ordering_and_refs(self) -> Trajectory:
        prev_ts: datetime | None = None
        seen_call_ids: set[str] = set()
        for ev in self.events:
            if prev_ts is not None and ev.timestamp < prev_ts:
                raise ValueError(
                    f"Events out of order in session {self.session_id}: "
                    f"{ev.event_id} earlier than predecessor"
                )
            prev_ts = ev.timestamp
            if isinstance(ev, ToolCallEvent):
                seen_call_ids.update(c.id for c in ev.tool_calls)
            elif isinstance(ev, ToolResultEvent):
                if ev.tool_call_id not in seen_call_ids:
                    # Allow it but flag — the validate stage will catch it.
                    self.tags.setdefault("dangling_tool_results", []).append(ev.event_id)
        return self

    def num_steps(self) -> int:
        """Number of assistant turns (text or tool-call) in the trajectory."""
        return sum(1 for e in self.events if isinstance(e, (AssistantEvent, ToolCallEvent)))

    def tool_set(self) -> list[str]:
        """Distinct tool names invoked, in first-seen order."""
        seen: list[str] = []
        for e in self.events:
            if isinstance(e, ToolCallEvent):
                for c in e.tool_calls:
                    if c.name not in seen:
                        seen.append(c.name)
        return seen

    def has_recovery(self) -> bool:
        """Heuristic: a tool error followed by a different tool call counts as recovery."""
        for i, ev in enumerate(self.events):
            if isinstance(ev, ToolResultEvent) and ev.is_error:
                for follow in self.events[i + 1 :]:
                    if isinstance(follow, ToolCallEvent):
                        return True
        return False

    def fingerprint(self) -> str:
        """Stable hash of user turns — useful for near-duplicate detection."""
        h = hashlib.sha256()
        for e in self.events:
            if isinstance(e, UserEvent):
                h.update(e.content.strip().lower().encode("utf-8"))
                h.update(b"\x00")
        return h.hexdigest()


class Session(BaseModel):
    """Wrapper around a trajectory plus session-level metadata.

    Useful when a session encapsulates multiple separable trajectories (e.g.,
    branching agent runs) — each trajectory keeps its own ordering invariants.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    trajectories: list[Trajectory] = Field(default_factory=list)
    user_id_hash: str | None = Field(
        default=None,
        description="Hashed user id; raw user ids must never appear here.",
    )
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

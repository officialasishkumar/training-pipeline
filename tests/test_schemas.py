"""Schema validators must enforce trajectory invariants."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)
from training_pipeline.schemas.exports import DPORecord, SFTMessage, SFTRecord


def _t(s: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).replace(second=s)


def test_trajectory_rejects_out_of_order_events():
    with pytest.raises(ValueError, match="out of order"):
        Trajectory(
            session_id="s1",
            events=[
                UserEvent(event_id="a", session_id="s1", timestamp=_t(2), content="hi"),
                UserEvent(event_id="b", session_id="s1", timestamp=_t(1), content="bye"),
            ],
        )


def test_trajectory_flags_dangling_tool_results():
    traj = Trajectory(
        session_id="s1",
        events=[
            UserEvent(event_id="u", session_id="s1", timestamp=_t(0), content="hi"),
            ToolResultEvent(
                event_id="r",
                session_id="s1",
                timestamp=_t(1),
                tool_call_id="missing",
                name="x",
                content="ok",
            ),
        ],
    )
    assert traj.tags["dangling_tool_results"] == ["r"]


def test_tool_call_event_requires_calls():
    with pytest.raises(ValidationError):
        ToolCallEvent(
            event_id="x",
            session_id="s",
            timestamp=_t(0),
            tool_calls=[],
        )


def test_trajectory_helpers(agent_trajectory: Trajectory):
    assert agent_trajectory.tool_set() == ["mandi_price"]
    assert agent_trajectory.has_recovery() is True
    # Two tool calls, one assistant text → 3 assistant turns.
    assert agent_trajectory.num_steps() == 3


def test_fingerprint_stable_across_whitespace():
    a = Trajectory(
        session_id="s1",
        events=[
            UserEvent(event_id="u", session_id="s1", timestamp=_t(0), content="Hello world  "),
        ],
    )
    b = Trajectory(
        session_id="s1",
        events=[
            UserEvent(event_id="u", session_id="s1", timestamp=_t(0), content="hello world"),
        ],
    )
    assert a.fingerprint() == b.fingerprint()


def test_sft_message_validators():
    with pytest.raises(ValidationError):
        SFTMessage(role="tool", content="x")  # missing tool_call_id
    with pytest.raises(ValidationError):
        SFTMessage(role="assistant")  # neither content nor tool_calls
    SFTMessage(role="user", content="hi")
    SFTMessage(role="tool", tool_call_id="c1", content="result")


def test_sft_record_minimum_messages():
    with pytest.raises(ValidationError):
        SFTRecord(messages=[SFTMessage(role="user", content="hi")])
    SFTRecord(
        messages=[
            SFTMessage(role="user", content="hi"),
            SFTMessage(role="assistant", content="hello"),
        ]
    )


def test_dpo_record_requires_assistant_in_completions():
    prompt = [SFTMessage(role="user", content="x")]
    with pytest.raises(ValidationError):
        DPORecord(
            prompt=prompt,
            chosen=[SFTMessage(role="user", content="y")],
            rejected=[SFTMessage(role="assistant", content="z")],
        )
    DPORecord(
        prompt=prompt,
        chosen=[SFTMessage(role="assistant", content="ok")],
        rejected=[SFTMessage(role="assistant", content="bad")],
    )

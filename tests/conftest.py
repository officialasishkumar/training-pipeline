"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)


def _ts(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, seconds, tzinfo=timezone.utc)


@pytest.fixture
def qa_trajectory() -> Trajectory:
    return Trajectory(
        session_id="qa-1",
        domain="agronomy",
        events=[
            UserEvent(
                event_id="u1",
                session_id="qa-1",
                timestamp=_ts(0),
                content="When to sow ragi?",
            ),
            AssistantEvent(
                event_id="a1",
                session_id="qa-1",
                timestamp=_ts(1),
                content="Ragi is sown in mid-June.",
            ),
        ],
    )


@pytest.fixture
def agent_trajectory() -> Trajectory:
    """Multi-tool trajectory with one error and a recovery."""
    return Trajectory(
        session_id="ag-1",
        domain="agronomy",
        events=[
            UserEvent(
                event_id="u1",
                session_id="ag-1",
                timestamp=_ts(0),
                content="Get tomato prices in Bengaluru and email me at user@example.com",
            ),
            ToolCallEvent(
                event_id="tc1",
                session_id="ag-1",
                timestamp=_ts(1),
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="mandi_price",
                        arguments={"commodity": "tomato", "market": "Bangalore"},
                    )
                ],
            ),
            ToolResultEvent(
                event_id="tr1",
                session_id="ag-1",
                timestamp=_ts(2),
                tool_call_id="c1",
                name="mandi_price",
                content='{"error": "market not found"}',
                is_error=True,
            ),
            ToolCallEvent(
                event_id="tc2",
                session_id="ag-1",
                timestamp=_ts(3),
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="mandi_price",
                        arguments={"commodity": "tomato", "market": "Bengaluru"},
                    )
                ],
            ),
            ToolResultEvent(
                event_id="tr2",
                session_id="ag-1",
                timestamp=_ts(4),
                tool_call_id="c2",
                name="mandi_price",
                content='{"min": 1200, "max": 1800}',
                is_error=False,
            ),
            AssistantEvent(
                event_id="a1",
                session_id="ag-1",
                timestamp=_ts(5),
                content="Tomatoes are at ₹1200-1800 per quintal.",
            ),
        ],
    )


@pytest.fixture
def openai_record() -> dict[str, Any]:
    return {
        "session_id": "oa-1",
        "messages": [
            {"role": "user", "content": "What's the moisture in plot A?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "soil_sensor",
                            "arguments": '{"plot": "A"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "soil_sensor",
                "content": '{"moisture": 0.32}',
            },
            {"role": "assistant", "content": "Plot A is at 32% moisture."},
        ],
    }


@pytest.fixture
def anthropic_record() -> dict[str, Any]:
    return {
        "session_id": "an-1",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the moisture?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "soil_sensor",
                        "input": {"plot": "A"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "soil_sensor",
                        "content": '{"moisture": 0.32}',
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "32%."}],
            },
        ],
    }


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    return tmp_path / "data.jsonl"

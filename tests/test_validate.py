"""Validation: tool registry, consistency, near-duplicate detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)
from training_pipeline.validate.consistency import (
    ToolRegistry,
    validate_consistency,
    validate_tool_call,
)
from training_pipeline.validate.splits import (
    detect_near_duplicates,
    split_integrity_report,
)


@pytest.fixture
def sample_registry(tmp_path: Path) -> ToolRegistry:
    p = tmp_path / "tools.yaml"
    p.write_text(
        "tools:\n  soil_sensor:\n    required: [plot]\n    properties:\n      plot: string\n",
        encoding="utf-8",
    )
    return ToolRegistry.from_yaml(p)


def test_unknown_tool_flagged(sample_registry: ToolRegistry):
    issues = validate_tool_call(
        ToolCall(id="c", name="not_a_tool", arguments={}),
        registry=sample_registry,
    )
    assert any(i.code == "UNKNOWN_TOOL" for i in issues)


def test_missing_required_field(sample_registry: ToolRegistry):
    issues = validate_tool_call(
        ToolCall(id="c", name="soil_sensor", arguments={}),
        registry=sample_registry,
    )
    assert any(i.code == "ARG_SCHEMA" for i in issues)


def test_wrong_arg_type(sample_registry: ToolRegistry):
    issues = validate_tool_call(
        ToolCall(id="c", name="soil_sensor", arguments={"plot": 123}),
        registry=sample_registry,
    )
    assert any(i.code == "ARG_SCHEMA" for i in issues)


def test_consistency_clean_trajectory(qa_trajectory: Trajectory):
    issues = validate_consistency(qa_trajectory)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []


def test_consistency_observation_contradiction():
    """Assistant claims success after a tool error → flagged warning."""
    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(event_id="u", session_id="s", content="run"),
            ToolCallEvent(
                event_id="tc",
                session_id="s",
                tool_calls=[ToolCall(id="c", name="x")],
            ),
            ToolResultEvent(
                event_id="tr",
                session_id="s",
                tool_call_id="c",
                name="x",
                content="boom",
                is_error=True,
            ),
            AssistantEvent(
                event_id="a",
                session_id="s",
                content="I successfully retrieved the data.",
            ),
        ],
    )
    issues = validate_consistency(traj)
    codes = {i.code for i in issues}
    assert "OBSERVATION_CONTRADICTION" in codes


def test_consistency_unresolved_call():
    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(event_id="u", session_id="s", content="run"),
            ToolCallEvent(
                event_id="tc",
                session_id="s",
                tool_calls=[ToolCall(id="c", name="x")],
            ),
        ],
    )
    issues = validate_consistency(traj)
    assert any(i.code == "UNRESOLVED_CALL" for i in issues)


def test_near_duplicate_within_set():
    a = Trajectory(
        session_id="s1",
        events=[
            UserEvent(
                event_id="u",
                session_id="s1",
                content="The quick brown fox jumps over the lazy dog.",
            )
        ],
    )
    b = Trajectory(
        session_id="s2",
        events=[
            UserEvent(
                event_id="u",
                session_id="s2",
                content="The quick brown fox jumps over the lazy dogs.",
            )
        ],
    )
    leaks = detect_near_duplicates([a, b], threshold=0.6)
    assert leaks


def test_split_integrity_no_leak_when_disjoint():
    a = Trajectory(
        session_id="a",
        events=[UserEvent(event_id="u", session_id="a", content="apple banana cherry")],
    )
    b = Trajectory(
        session_id="b",
        events=[
            UserEvent(event_id="u", session_id="b", content="completely different content here")
        ],
    )
    report = split_integrity_report({"train": [a], "val": [b]}, threshold=0.85)
    assert report["total_leaks"] == 0


def test_split_integrity_flags_cross_split_leak():
    text = "rice paddy yield BPT 5204"
    a = Trajectory(
        session_id="a",
        events=[UserEvent(event_id="u", session_id="a", content=text)],
    )
    b = Trajectory(
        session_id="b",
        events=[UserEvent(event_id="u", session_id="b", content=text)],
    )
    report = split_integrity_report({"train": [a], "val": [b]}, threshold=0.85)
    assert report["total_leaks"] >= 1

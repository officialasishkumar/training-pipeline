"""Trajectory tagging and stratified split."""

from __future__ import annotations

import pytest

from training_pipeline.schemas.events import (
    Trajectory,
)
from training_pipeline.tagging.complexity import (
    classify_complexity,
    compute_tags,
    tag_trajectory,
)
from training_pipeline.tagging.stratify import stratified_split, stratum_key


def test_qa_trajectory_is_easy_or_below(qa_trajectory: Trajectory):
    tags = compute_tags(qa_trajectory)
    assert tags.complexity_band in ("trivial", "easy")
    assert tags.has_recovery is False


def test_agent_trajectory_recovers(agent_trajectory: Trajectory):
    tags = compute_tags(agent_trajectory)
    assert tags.has_recovery is True
    assert tags.num_tool_calls == 2
    assert tags.distinct_tools == 1
    assert tags.complexity_band in ("medium", "hard", "extreme")


def test_classify_band_boundaries():
    assert classify_complexity(0.0) == "trivial"
    assert classify_complexity(2.0) == "easy"
    assert classify_complexity(3.5) == "medium"
    assert classify_complexity(6.0) == "hard"
    assert classify_complexity(8.0) == "extreme"


def test_tag_trajectory_attaches_complexity(agent_trajectory: Trajectory):
    out = tag_trajectory(agent_trajectory)
    assert "complexity" in out.tags
    assert "complexity_band" in out.tags["complexity"]


def test_stratum_key_uses_complexity_and_domain(qa_trajectory: Trajectory):
    tagged = tag_trajectory(qa_trajectory)
    key = stratum_key(tagged)
    assert "agronomy" in key


def test_stratified_split_balances_strata(qa_trajectory: Trajectory, agent_trajectory: Trajectory):
    qa_tagged = tag_trajectory(qa_trajectory)
    ag_tagged = tag_trajectory(agent_trajectory)

    # 20 trajectories: 10 easy, 10 hard.
    items = []
    for i in range(10):
        items.append(qa_tagged.model_copy(update={"session_id": f"q-{i}"}))
        items.append(ag_tagged.model_copy(update={"session_id": f"a-{i}"}))

    s = stratified_split(items, fractions=(0.7, 0.15, 0.15), seed=1)
    assert len(s.train) + len(s.val) + len(s.test) == len(items)
    # Both strata should have at least one item in train.
    train_strata = {stratum_key(items[i]) for i in s.train}
    assert len(train_strata) == 2


def test_stratified_split_is_deterministic(qa_trajectory: Trajectory):
    tagged = tag_trajectory(qa_trajectory)
    items = [tagged.model_copy(update={"session_id": f"q-{i}"}) for i in range(20)]
    s1 = stratified_split(items, seed=42)
    s2 = stratified_split(items, seed=42)
    assert s1.train == s2.train
    assert s1.val == s2.val


def test_stratified_split_validates_fractions():
    with pytest.raises(ValueError):
        stratified_split([], fractions=(0.5, 0.4, 0.0), seed=0)  # sums to 0.9


def test_repair_loop_zero_for_clean_trajectory(qa_trajectory: Trajectory):
    tags = compute_tags(qa_trajectory)
    assert tags.repair_loop_depth == 0
    assert tags.thrashing is False


def test_repair_loop_one_after_single_recovery(agent_trajectory: Trajectory):
    """One error → retry that succeeds = depth 1, no thrashing."""
    tags = compute_tags(agent_trajectory)
    assert tags.repair_loop_depth == 1
    assert tags.thrashing is False


def test_repair_loop_thrashing_after_repeated_errors():
    """Three same-tool errors in a row should report depth=3 and thrashing=True."""
    from datetime import datetime, timezone

    from training_pipeline.schemas.events import (
        ToolCall,
        ToolCallEvent,
        ToolResultEvent,
        Trajectory,
        UserEvent,
    )

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def call(i: int) -> ToolCallEvent:
        return ToolCallEvent(
            event_id=f"tc{i}",
            session_id="s",
            timestamp=base.replace(second=2 * i),
            tool_calls=[ToolCall(id=f"c{i}", name="flaky_tool")],
        )

    def result(i: int, *, error: bool) -> ToolResultEvent:
        return ToolResultEvent(
            event_id=f"tr{i}",
            session_id="s",
            timestamp=base.replace(second=2 * i + 1),
            tool_call_id=f"c{i}",
            name="flaky_tool",
            content="boom" if error else '{"ok": true}',
            is_error=error,
        )

    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(event_id="u", session_id="s", timestamp=base, content="run"),
            call(0),
            result(0, error=True),
            call(1),
            result(1, error=True),
            call(2),
            result(2, error=True),
            call(3),
            result(3, error=False),
        ],
    )
    tags = compute_tags(traj)
    assert tags.repair_loop_depth == 3
    assert tags.thrashing is True
    assert tags.complexity_band in ("hard", "extreme")


def test_repair_loop_resets_on_tool_switch():
    """Switching to a different tool resets the repair streak."""
    from datetime import datetime, timezone

    from training_pipeline.schemas.events import (
        ToolCall,
        ToolCallEvent,
        ToolResultEvent,
        Trajectory,
        UserEvent,
    )

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(event_id="u", session_id="s", timestamp=base, content="run"),
            ToolCallEvent(
                event_id="tc1",
                session_id="s",
                timestamp=base.replace(second=1),
                tool_calls=[ToolCall(id="c1", name="tool_a")],
            ),
            ToolResultEvent(
                event_id="tr1",
                session_id="s",
                timestamp=base.replace(second=2),
                tool_call_id="c1",
                name="tool_a",
                content="boom",
                is_error=True,
            ),
            ToolCallEvent(
                event_id="tc2",
                session_id="s",
                timestamp=base.replace(second=3),
                tool_calls=[ToolCall(id="c2", name="tool_b")],
            ),
            ToolResultEvent(
                event_id="tr2",
                session_id="s",
                timestamp=base.replace(second=4),
                tool_call_id="c2",
                name="tool_b",
                content="ok",
                is_error=False,
            ),
        ],
    )
    tags = compute_tags(traj)
    assert tags.repair_loop_depth == 0  # switched tools, didn't retry same one
    assert tags.thrashing is False

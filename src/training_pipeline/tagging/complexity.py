"""Compute compositional-complexity, recovery, and ambiguity tags.

These tags are *the* knob trainers use to:

- stage curricula (start on easy single-tool prompts, ramp to multi-tool)
- mix preference data with SFT data at appropriate ratios
- decide which trajectories a smaller student can plausibly imitate

Heuristics here are intentionally simple and explainable. Replace with a
loss-based hardness score (see ``hardness.py`` placeholder) once a reference
model is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)

ComplexityBand = Literal["trivial", "easy", "medium", "hard", "extreme"]


@dataclass
class ComplexityTags:
    num_user_turns: int
    num_assistant_turns: int
    num_tool_calls: int
    num_tool_errors: int
    distinct_tools: int
    tool_set: list[str]
    has_recovery: bool
    has_dangling_tool_results: bool
    ambiguity_score: float
    """0.0 = clear request, 1.0 = highly ambiguous (heuristic)."""
    repair_loop_depth: int
    """Largest run of consecutive errors on the *same* tool name, capturing the
    worst stretch where the agent kept retrying without success. 0 means no
    repeats; 1 = one error followed by recovery; >=2 means a stuck loop."""
    thrashing: bool
    """True when the agent retried the same tool with an error >=2 times in a
    row. Distinct from ``has_recovery`` (which can be a single retry that
    succeeds) — thrashing signals the agent failed to find a path forward
    quickly. Useful as a hardness signal and for filtering low-quality
    behaviour-cloning examples."""
    complexity_band: ComplexityBand
    complexity_score: float
    """Continuous score 0..10 — input to band classification."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_user_turns": self.num_user_turns,
            "num_assistant_turns": self.num_assistant_turns,
            "num_tool_calls": self.num_tool_calls,
            "num_tool_errors": self.num_tool_errors,
            "distinct_tools": self.distinct_tools,
            "tool_set": self.tool_set,
            "has_recovery": self.has_recovery,
            "has_dangling_tool_results": self.has_dangling_tool_results,
            "ambiguity_score": round(self.ambiguity_score, 3),
            "repair_loop_depth": self.repair_loop_depth,
            "thrashing": self.thrashing,
            "complexity_band": self.complexity_band,
            "complexity_score": round(self.complexity_score, 3),
        }


_AMBIGUITY_HINTS: tuple[str, ...] = (
    "?",
    "could you",
    "what about",
    "or maybe",
    "not sure",
    "either",
    "perhaps",
    "kind of",
    "sort of",
    "depending on",
    "if possible",
    "i'm not sure",
)


def _ambiguity_score(trajectory: Trajectory) -> float:
    """Cheap proxy: count ambiguity markers normalized by user turn length."""
    user_text = " ".join(e.content.lower() for e in trajectory.events if isinstance(e, UserEvent))
    if not user_text:
        return 0.0
    hits = sum(user_text.count(h) for h in _AMBIGUITY_HINTS)
    # log scale so 1 hint != 5x as ambiguous as 0
    return min(1.0, math.log1p(hits) / math.log1p(5))


def classify_complexity(score: float) -> ComplexityBand:
    if score < 1.0:
        return "trivial"
    if score < 2.5:
        return "easy"
    if score < 4.5:
        return "medium"
    if score < 7.0:
        return "hard"
    return "extreme"


def _repair_loop_depth(trajectory: Trajectory) -> int:
    """Length of the longest consecutive same-tool error streak.

    We walk the events in order and pair each tool call with its result via
    ``tool_call_id``. A streak counts a *new* call to a tool whose previous
    call (with the same name) errored. Calls to a different tool reset the
    streak — that's the agent's choice to switch strategy.
    """
    last_call_name_by_id: dict[str, str] = {}
    last_streak_tool: str | None = None
    current_streak = 0
    max_streak = 0
    last_was_error: dict[str, bool] = {}

    # First pass: index call name and (later) error state by call id.
    for ev in trajectory.events:
        if isinstance(ev, ToolCallEvent):
            for c in ev.tool_calls:
                last_call_name_by_id[c.id] = c.name
        elif isinstance(ev, ToolResultEvent):
            last_was_error[ev.tool_call_id] = ev.is_error

    # Second pass: walk forward looking for retry streaks.
    pending_error_tool: str | None = None
    for ev in trajectory.events:
        if isinstance(ev, ToolCallEvent):
            for c in ev.tool_calls:
                if pending_error_tool == c.name:
                    if last_streak_tool == c.name:
                        current_streak += 1
                    else:
                        current_streak = 1
                        last_streak_tool = c.name
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
                    last_streak_tool = None
                pending_error_tool = None
        elif isinstance(ev, ToolResultEvent):
            tool_name = last_call_name_by_id.get(ev.tool_call_id, ev.name)
            pending_error_tool = tool_name if ev.is_error else None
    return max_streak


def compute_tags(trajectory: Trajectory) -> ComplexityTags:
    n_user = sum(1 for e in trajectory.events if isinstance(e, UserEvent))
    n_assist = sum(1 for e in trajectory.events if isinstance(e, AssistantEvent))
    tool_calls = [e for e in trajectory.events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in trajectory.events if isinstance(e, ToolResultEvent)]
    n_calls = sum(len(e.tool_calls) for e in tool_calls)
    n_errors = sum(1 for r in tool_results if r.is_error)
    tools = trajectory.tool_set()
    distinct = len(tools)
    recovery = trajectory.has_recovery()
    dangling = bool(trajectory.tags.get("dangling_tool_results"))

    ambiguity = _ambiguity_score(trajectory)
    repair_depth = _repair_loop_depth(trajectory)
    thrashing = repair_depth >= 2

    # Continuous score — designed so simple Q&A is < 1, multi-tool with recovery > 5.
    score = (
        max(0, n_user - 1) * 0.6
        + n_calls * 0.5
        + distinct * 0.7
        + (2.0 if recovery else 0.0)
        + n_errors * 0.4
        + ambiguity * 1.5
        + (1.0 if dangling else 0.0)
        + repair_depth * 0.8
        + (1.5 if thrashing else 0.0)
    )

    return ComplexityTags(
        num_user_turns=n_user,
        num_assistant_turns=n_assist + len(tool_calls),
        num_tool_calls=n_calls,
        num_tool_errors=n_errors,
        distinct_tools=distinct,
        tool_set=tools,
        has_recovery=recovery,
        has_dangling_tool_results=dangling,
        ambiguity_score=ambiguity,
        repair_loop_depth=repair_depth,
        thrashing=thrashing,
        complexity_band=classify_complexity(score),
        complexity_score=score,
    )


def tag_trajectory(trajectory: Trajectory) -> Trajectory:
    """Attach computed tags under ``trajectory.tags['complexity']``."""
    tags = compute_tags(trajectory)
    new_tags = {**trajectory.tags, "complexity": tags.as_dict()}
    return trajectory.model_copy(update={"tags": new_tags})

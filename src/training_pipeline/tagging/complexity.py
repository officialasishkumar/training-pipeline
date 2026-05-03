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
from dataclasses import dataclass, field
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
    user_text = " ".join(
        e.content.lower() for e in trajectory.events if isinstance(e, UserEvent)
    )
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

    # Continuous score — designed so simple Q&A is < 1, multi-tool with recovery > 5.
    score = (
        max(0, n_user - 1) * 0.6
        + n_calls * 0.5
        + distinct * 0.7
        + (2.0 if recovery else 0.0)
        + n_errors * 0.4
        + ambiguity * 1.5
        + (1.0 if dangling else 0.0)
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
        complexity_band=classify_complexity(score),
        complexity_score=score,
    )


def tag_trajectory(trajectory: Trajectory) -> Trajectory:
    """Attach computed tags under ``trajectory.tags['complexity']``."""
    tags = compute_tags(trajectory)
    new_tags = {**trajectory.tags, "complexity": tags.as_dict()}
    return trajectory.model_copy(update={"tags": new_tags})

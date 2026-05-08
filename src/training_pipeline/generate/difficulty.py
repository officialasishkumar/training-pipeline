"""Difficulty tier and edge-case classification for synthetic trajectories.

The mentor's framing — "assume you have enough volume, not enough quality
or edge cases" — means every trajectory needs to be labelled along two
axes downstream:

* ``DifficultyTier``: ``easy`` / ``medium`` / ``hard``. Drives curriculum
  ordering and SFT/DPO mix ratios.
* ``EdgeCase``: a small, *non-exclusive* set of category flags. A
  trajectory can be both ``multi_tool`` and ``tool_failure_recovery``;
  ``jailbreak_refusal`` is independent of the rest.

Both signals reuse the existing complexity tags when available so the
classifier is cheap (no extra model calls). They also fall back to direct
trajectory inspection so generation can label rows before they hit the
``tag`` stage.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)
from training_pipeline.tagging.complexity import compute_tags


class DifficultyTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EdgeCase(str, Enum):
    PURE_QA = "pure_qa"
    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    TOOL_FAILURE_RECOVERY = "tool_failure_recovery"
    AMBIGUOUS_QUERY = "ambiguous_query"
    MULTILINGUAL = "multilingual"
    JAILBREAK_REFUSAL = "jailbreak_refusal"


@dataclass(frozen=True)
class DifficultyAssessment:
    tier: DifficultyTier
    edge_cases: frozenset[EdgeCase]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "edge_cases": sorted(c.value for c in self.edge_cases),
            "score": round(self.score, 3),
        }


# Devanagari + Tamil + Telugu + Kannada + Malayalam + Bengali + Gurmukhi.
_INDIC_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)

# Shortlisted English markers that appear in clear jailbreak attempts. We
# *don't* flag every "ignore" — a real classifier should replace this. The
# heuristic exists to give a non-zero baseline.
_JAILBREAK_PATTERNS = (
    re.compile(r"\bignore (all |the )?(previous|above) (instructions|rules)\b", re.I),
    re.compile(r"\b(act as|pretend to be|roleplay as) (?!a (farmer|user))", re.I),
    re.compile(r"\bdo anything now\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bbypass (the )?(safety|filter|guardrails?)\b", re.I),
    re.compile(r"\bdeveloper mode\b", re.I),
)

_REFUSAL_PATTERNS = (
    re.compile(r"\bi (can(?:'|no)t|won't|cannot)\b", re.I),
    re.compile(r"\b(against|outside) (my|the) (policy|guidelines)\b", re.I),
    re.compile(r"\bi'm not able to\b", re.I),
    re.compile(r"\bi must decline\b", re.I),
)


def _has_indic(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _INDIC_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _user_text(traj: Trajectory) -> str:
    return " ".join(e.content for e in traj.events if isinstance(e, UserEvent))


def _assistant_text(traj: Trajectory) -> str:
    return " ".join(e.content for e in traj.events if isinstance(e, AssistantEvent))


def _existing_complexity(traj: Trajectory) -> dict[str, Any]:
    tags = traj.tags or {}
    if isinstance(tags, dict):
        comp = tags.get("complexity")
        if isinstance(comp, dict):
            return comp
    return {}


def _complexity_signals(traj: Trajectory) -> dict[str, Any]:
    """Use cached complexity tags when present, otherwise compute on demand."""
    cached = _existing_complexity(traj)
    if cached:
        return cached
    return compute_tags(traj).as_dict()


def assign_difficulty(traj: Trajectory) -> DifficultyTier:
    """Map a trajectory's complexity signals onto a 3-tier scale.

    The existing complexity classifier has 5 bands; for curriculum mixing
    we collapse to 3 because that's the granularity SFT/DPO ratios are
    actually tuned at.
    """
    return _difficulty_from_signals(_complexity_signals(traj))


def _difficulty_from_signals(signals: dict[str, Any]) -> DifficultyTier:
    score = float(signals.get("complexity_score", 0.0))
    n_calls = int(signals.get("num_tool_calls", 0))
    n_errors = int(signals.get("num_tool_errors", 0))
    has_recovery = bool(signals.get("has_recovery", False))
    distinct = int(signals.get("distinct_tools", 0))

    if (
        n_errors >= 2
        or (has_recovery and n_calls >= 3)
        or distinct >= 3
        or score >= 6.5
    ):
        return DifficultyTier.HARD
    if n_calls >= 2 or has_recovery or score >= 3.0 or distinct >= 2:
        return DifficultyTier.MEDIUM
    return DifficultyTier.EASY


def flag_edge_cases(traj: Trajectory) -> frozenset[EdgeCase]:
    flags: set[EdgeCase] = set()
    n_calls = sum(1 for e in traj.events if isinstance(e, ToolCallEvent))
    distinct_tools = len(
        {
            c.name
            for e in traj.events
            if isinstance(e, ToolCallEvent)
            for c in e.tool_calls
        }
    )
    n_errors = sum(1 for e in traj.events if isinstance(e, ToolResultEvent) and e.is_error)
    has_recovery = traj.has_recovery()

    if n_calls == 0:
        flags.add(EdgeCase.PURE_QA)
    elif distinct_tools <= 1:
        flags.add(EdgeCase.SINGLE_TOOL)
    if distinct_tools >= 2:
        flags.add(EdgeCase.MULTI_TOOL)
    if has_recovery or n_errors >= 1:
        flags.add(EdgeCase.TOOL_FAILURE_RECOVERY)

    user_text = _user_text(traj)
    if user_text:
        ambiguity = float(_complexity_signals(traj).get("ambiguity_score", 0.0))
        # ``ambiguity_score`` is normalized 0..1; 0.4 is the empirical floor
        # where ambiguity markers cluster (multiple "or"/"maybe"/"perhaps").
        if ambiguity >= 0.4:
            flags.add(EdgeCase.AMBIGUOUS_QUERY)

        if _has_indic(user_text):
            flags.add(EdgeCase.MULTILINGUAL)

        if any(p.search(user_text) for p in _JAILBREAK_PATTERNS):
            assistant_text = _assistant_text(traj)
            if any(p.search(assistant_text) for p in _REFUSAL_PATTERNS):
                flags.add(EdgeCase.JAILBREAK_REFUSAL)

    return frozenset(flags)


def assess(traj: Trajectory) -> DifficultyAssessment:
    """Combined difficulty + edge-case classification."""
    signals = _complexity_signals(traj)
    return DifficultyAssessment(
        tier=_difficulty_from_signals(signals),
        edge_cases=flag_edge_cases(traj),
        score=float(signals.get("complexity_score", 0.0)),
    )


def annotate(traj: Trajectory) -> Trajectory:
    """Attach a ``difficulty`` block under ``trajectory.tags``."""
    a = assess(traj)
    new_tags = {**traj.tags, "difficulty": a.as_dict()}
    return traj.model_copy(update={"tags": new_tags})


def stratify(
    trajectories: Iterable[Trajectory],
    *,
    cap_per_bucket: int | None = None,
) -> list[Trajectory]:
    """Cap-per-bucket sampler over (tier x edge_case_signature).

    The mentor wants diverse data, not the long tail of "easy/single_tool".
    This is a deliberately small sampler: bucket every trajectory by its
    tier plus the sorted set of its edge cases, then keep at most
    ``cap_per_bucket`` from each bucket.
    """
    buckets: dict[tuple[str, tuple[str, ...]], list[Trajectory]] = {}
    for traj in trajectories:
        a = assess(traj)
        key = (a.tier.value, tuple(sorted(c.value for c in a.edge_cases)))
        buckets.setdefault(key, []).append(traj)
    out: list[Trajectory] = []
    for items in buckets.values():
        if cap_per_bucket is not None and cap_per_bucket >= 0:
            out.extend(items[:cap_per_bucket])
        else:
            out.extend(items)
    return out

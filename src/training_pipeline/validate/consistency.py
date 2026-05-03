"""Tool-call/observation consistency validation.

Three classes of issue we catch here:

1. **Schema** — a tool call's arguments don't match the registered schema
   (unknown tool, missing required field, wrong type).
2. **Reference** — a ``ToolResultEvent`` doesn't reference a prior call, or a
   call has no result.
3. **Observation contradiction** — heuristic: the assistant's textual
   continuation contradicts the result (tool reported error but assistant says
   "I successfully retrieved..."). Conservative — flag, don't drop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsistencyIssue:
    severity: str  # "warning" | "error"
    code: str
    message: str
    event_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    required: tuple[str, ...] = ()
    properties: dict[str, str] = field(default_factory=dict)
    """Property name → JSON-Schema-ish type ('string', 'number', 'boolean', 'object', 'array', 'any')."""
    description: str = ""

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        """Return error strings; empty if valid."""
        errs: list[str] = []
        if not isinstance(args, dict):
            return [f"arguments are not a dict (got {type(args).__name__})"]
        for r in self.required:
            if r not in args:
                errs.append(f"missing required field {r!r}")
        for k, v in args.items():
            expected = self.properties.get(k, "any")
            if expected == "any":
                continue
            if not _matches_type(v, expected):
                errs.append(f"field {k!r} expected {expected}, got {type(v).__name__}")
        return errs


def _matches_type(v: Any, expected: str) -> bool:
    return {
        "string": isinstance(v, str),
        "number": isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": isinstance(v, int) and not isinstance(v, bool),
        "boolean": isinstance(v, bool),
        "object": isinstance(v, dict),
        "array": isinstance(v, list),
        "null": v is None,
    }.get(expected, True)


@dataclass
class ToolRegistry:
    """Loadable map of tool name → spec.

    A YAML file looks like:

    .. code-block:: yaml

        tools:
          soil_sensor:
            description: Read soil moisture for a plot
            required: [plot]
            properties:
              plot: string
              depth_cm: integer
    """

    tools: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ToolRegistry:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        tools_raw = data.get("tools", {}) or {}
        tools = {
            name: ToolSpec(
                name=name,
                required=tuple(spec.get("required", []) or []),
                properties=dict(spec.get("properties", {}) or {}),
                description=str(spec.get("description", "")),
            )
            for name, spec in tools_raw.items()
        }
        return cls(tools=tools)

    def has(self, name: str) -> bool:
        return name in self.tools

    def __bool__(self) -> bool:
        return bool(self.tools)


def validate_tool_call(
    call: ToolCall, registry: ToolRegistry | None = None
) -> list[ConsistencyIssue]:
    """Check a single tool call against an optional registry."""
    issues: list[ConsistencyIssue] = []
    if registry and registry.tools:
        if not registry.has(call.name):
            issues.append(
                ConsistencyIssue(
                    severity="error",
                    code="UNKNOWN_TOOL",
                    message=f"Tool {call.name!r} not in registry",
                    event_id=None,
                    extra={"tool_name": call.name},
                )
            )
        else:
            spec = registry.tools[call.name]
            for err in spec.validate_args(call.arguments):
                issues.append(
                    ConsistencyIssue(
                        severity="error",
                        code="ARG_SCHEMA",
                        message=f"{call.name}: {err}",
                        event_id=None,
                        extra={"tool_name": call.name},
                    )
                )
    return issues


_SUCCESS_PHRASES = (
    "successfully",
    "i was able to",
    "the answer is",
    "here is",
    "here are",
    "i found",
    "i retrieved",
    "got the",
)
_FAILURE_PHRASES = (
    "i couldn't",
    "i could not",
    "failed",
    "error",
    "unable to",
    "no results",
    "didn't work",
    "something went wrong",
    "let me try",
    "try again",
)


def validate_consistency(
    trajectory: Trajectory,
    *,
    registry: ToolRegistry | None = None,
) -> list[ConsistencyIssue]:
    """Run all consistency checks for a trajectory.

    The checks are read-only; the caller decides whether to drop or just flag.
    """
    issues: list[ConsistencyIssue] = []

    pending_calls: dict[str, ToolCall] = {}
    seen_results: set[str] = set()
    last_result: ToolResultEvent | None = None

    for ev in trajectory.events:
        if isinstance(ev, ToolCallEvent):
            for call in ev.tool_calls:
                pending_calls[call.id] = call
                issues.extend(
                    [_with_event(i, ev.event_id) for i in validate_tool_call(call, registry)]
                )
        elif isinstance(ev, ToolResultEvent):
            seen_results.add(ev.tool_call_id)
            last_result = ev
            if ev.tool_call_id not in pending_calls:
                issues.append(
                    ConsistencyIssue(
                        severity="error",
                        code="DANGLING_RESULT",
                        message=(
                            f"Result {ev.event_id} references tool_call_id "
                            f"{ev.tool_call_id!r} with no preceding call"
                        ),
                        event_id=ev.event_id,
                    )
                )
            else:
                expected_name = pending_calls[ev.tool_call_id].name
                if ev.name != expected_name:
                    issues.append(
                        ConsistencyIssue(
                            severity="warning",
                            code="NAME_MISMATCH",
                            message=(
                                f"Result names {ev.name!r} but call was {expected_name!r}"
                            ),
                            event_id=ev.event_id,
                        )
                    )
                # Try to parse content as JSON; surface obvious garbage.
                try:
                    json.loads(ev.content)
                except (ValueError, TypeError):
                    pass  # tools often return plain text — not an issue
        elif isinstance(ev, AssistantEvent) and last_result is not None:
            text = ev.content.lower()
            said_success = any(p in text for p in _SUCCESS_PHRASES)
            said_failure = any(p in text for p in _FAILURE_PHRASES)
            if last_result.is_error and said_success and not said_failure:
                issues.append(
                    ConsistencyIssue(
                        severity="warning",
                        code="OBSERVATION_CONTRADICTION",
                        message=(
                            "Assistant claims success after a tool reported an error"
                        ),
                        event_id=ev.event_id,
                        extra={"related_result": last_result.event_id},
                    )
                )
            if not last_result.is_error and said_failure and not said_success:
                issues.append(
                    ConsistencyIssue(
                        severity="warning",
                        code="OBSERVATION_CONTRADICTION",
                        message=(
                            "Assistant claims failure after a tool returned without error"
                        ),
                        event_id=ev.event_id,
                        extra={"related_result": last_result.event_id},
                    )
                )
            last_result = None  # only check the assistant turn directly after

    # Calls without results
    unresolved = set(pending_calls) - seen_results
    for call_id in unresolved:
        issues.append(
            ConsistencyIssue(
                severity="warning",
                code="UNRESOLVED_CALL",
                message=f"Tool call {call_id!r} has no matching result",
            )
        )

    return issues


def _with_event(issue: ConsistencyIssue, event_id: str) -> ConsistencyIssue:
    if issue.event_id:
        return issue
    return ConsistencyIssue(
        severity=issue.severity,
        code=issue.code,
        message=issue.message,
        event_id=event_id,
        extra=issue.extra,
    )

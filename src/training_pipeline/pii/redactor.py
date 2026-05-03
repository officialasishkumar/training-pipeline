"""Apply detections back to text and trajectories.

Two important properties:

1. **Consistency** — the same value gets the same placeholder *within a
   trajectory* so multi-turn coreference is preserved. ``user@example.com``
   stays as ``[EMAIL_1]`` for the whole trajectory; a different email becomes
   ``[EMAIL_2]``.
2. **Auditability** — every redaction is recorded so a sampling tool can
   surface raw values for human review *without* persisting them in the main
   output.

For tool arguments and results we redact the JSON-serialised form; downstream
trainers tokenise the same string anyway.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import orjson

from training_pipeline.pii.rules import (
    BUILTIN_RULES,
    PIIDetection,
    PIIRule,
    detect_all,
)
from training_pipeline.schemas.events import (
    AssistantEvent,
    ErrorEvent,
    Event,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)

log = logging.getLogger(__name__)


@dataclass
class _RedactionLog:
    """Internal: detections recorded during redaction, used by the audit sampler."""

    text: str
    detections: list[PIIDetection]
    placeholders: dict[str, str]


@dataclass
class RedactionResult:
    """Return value of ``Redactor.redact_trajectory``.

    Attributes:
        trajectory: the redacted trajectory (a new object; input is unchanged).
        report: per-category counts, useful for an audit summary.
        sample: a list of (event_id, original_text, redacted_text, detections)
            tuples — *only* populated when ``audit_rate`` is positive.
    """

    trajectory: Trajectory
    report: dict[str, int] = field(default_factory=dict)
    sample: list[dict[str, Any]] = field(default_factory=list)


class Redactor:
    """Stateful redactor: assigns stable placeholder ids per trajectory."""

    def __init__(
        self,
        rules: tuple[PIIRule, ...] = BUILTIN_RULES,
        *,
        consistent_placeholders: bool = True,
    ) -> None:
        self.rules = rules
        self.consistent_placeholders = consistent_placeholders

    def redact_text(
        self,
        text: str,
        *,
        memo: dict[str, str] | None = None,
        category_counts: dict[str, int] | None = None,
    ) -> tuple[str, list[PIIDetection]]:
        """Redact text and return ``(redacted, detections)``.

        ``memo`` maps original-value to placeholder for consistency across the
        trajectory; pass the same dict between calls. ``category_counts`` keeps
        the running total per category.
        """
        if memo is None:
            memo = {}
        if category_counts is None:
            category_counts = defaultdict(int)
        detections = detect_all(text, self.rules)
        if not detections:
            return text, []
        # Resolve overlaps: keep the leftmost-longest match.
        resolved: list[PIIDetection] = []
        for d in detections:
            if resolved and d.start < resolved[-1].end:
                continue
            resolved.append(d)
        # Build replaced text in one pass.
        out: list[str] = []
        cursor = 0
        for d in resolved:
            out.append(text[cursor : d.start])
            placeholder = self._placeholder(d, memo, category_counts)
            out.append(placeholder)
            cursor = d.end
        out.append(text[cursor:])
        return "".join(out), resolved

    def _placeholder(
        self,
        d: PIIDetection,
        memo: dict[str, str],
        category_counts: dict[str, int],
    ) -> str:
        """Stable placeholder for a single detection."""
        if self.consistent_placeholders and d.text in memo:
            return memo[d.text]
        category_counts[d.category] = category_counts.get(d.category, 0) + 1
        n = sum(1 for v in memo.values() if v.startswith(f"[{d.category}_"))
        token = f"[{d.category}_{n + 1}]"
        memo[d.text] = token
        return token

    def redact_trajectory(
        self,
        trajectory: Trajectory,
        *,
        record_for_audit: bool = False,
    ) -> RedactionResult:
        """Redact every textual field of every event."""
        memo: dict[str, str] = {}
        category_counts: dict[str, int] = defaultdict(int)
        new_events: list[Event] = []
        sample: list[dict[str, Any]] = []
        for ev in trajectory.events:
            new_ev, ev_detections = self._redact_event(
                ev, memo=memo, category_counts=category_counts
            )
            new_events.append(new_ev)
            if record_for_audit and ev_detections:
                sample.append(
                    {
                        "event_id": ev.event_id,
                        "session_id": ev.session_id,
                        "detections": [
                            {
                                "rule": d.rule,
                                "category": d.category,
                                "text": d.text,
                            }
                            for d in ev_detections
                        ],
                    }
                )

        new_traj = Trajectory(
            session_id=trajectory.session_id,
            events=new_events,
            source=trajectory.source,
            domain=trajectory.domain,
            tags={**trajectory.tags, "pii_redacted": True, "pii_counts": dict(category_counts)},
            schema_version=trajectory.schema_version,
        )
        return RedactionResult(
            trajectory=new_traj,
            report=dict(category_counts),
            sample=sample,
        )

    def _redact_event(
        self,
        ev: Event,
        *,
        memo: dict[str, str],
        category_counts: dict[str, int],
    ) -> tuple[Event, list[PIIDetection]]:
        all_detections: list[PIIDetection] = []
        if isinstance(ev, UserEvent):
            new_text, det = self.redact_text(ev.content, memo=memo, category_counts=category_counts)
            all_detections.extend(det)
            return ev.model_copy(update={"content": new_text}), all_detections
        if isinstance(ev, AssistantEvent):
            new_text, det = self.redact_text(ev.content, memo=memo, category_counts=category_counts)
            all_detections.extend(det)
            return ev.model_copy(update={"content": new_text}), all_detections
        if isinstance(ev, ToolCallEvent):
            new_calls: list[ToolCall] = []
            for call in ev.tool_calls:
                args_text = orjson.dumps(call.arguments).decode("utf-8")
                redacted_args_text, det = self.redact_text(
                    args_text, memo=memo, category_counts=category_counts
                )
                all_detections.extend(det)
                try:
                    new_args = orjson.loads(redacted_args_text)
                    if not isinstance(new_args, dict):
                        new_args = {"_redacted": redacted_args_text}
                except orjson.JSONDecodeError:
                    new_args = {"_redacted": redacted_args_text}
                new_calls.append(ToolCall(id=call.id, name=call.name, arguments=new_args))
            return ev.model_copy(update={"tool_calls": new_calls}), all_detections
        if isinstance(ev, ToolResultEvent):
            new_text, det = self.redact_text(ev.content, memo=memo, category_counts=category_counts)
            all_detections.extend(det)
            return ev.model_copy(update={"content": new_text}), all_detections
        if isinstance(ev, ErrorEvent):
            new_msg, det = self.redact_text(ev.message, memo=memo, category_counts=category_counts)
            all_detections.extend(det)
            return ev.model_copy(update={"message": new_msg}), all_detections
        return ev, all_detections


def redact_trajectory(
    trajectory: Trajectory,
    *,
    rules: tuple[PIIRule, ...] = BUILTIN_RULES,
    record_for_audit: bool = False,
) -> RedactionResult:
    """Convenience: stateless wrapper around ``Redactor.redact_trajectory``."""
    return Redactor(rules=rules).redact_trajectory(trajectory, record_for_audit=record_for_audit)

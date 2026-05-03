"""DPO export: build prompt/chosen/rejected pairs from trajectories.

Three pair-construction strategies, configurable per run:

- ``feedback`` — read explicit feedback from trajectory ``tags`` (e.g. user
  thumbs-up/down on alternative responses). Best signal, lowest volume.
- ``failure_recovery`` — within an agent trajectory, take the assistant turn
  *after* a tool error (recovery) as ``chosen`` and the contradicting turn
  *before* the recovery as ``rejected``.
- ``synthetic`` — use a registered generator function to produce a "rejected"
  by perturbing the chosen (placeholder, no built-in generator).

The schema is invariant across strategies — only ``metadata.source`` differs.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from training_pipeline.export.sft import trajectory_to_messages
from training_pipeline.ingest.parsers import write_jsonl
from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
)
from training_pipeline.schemas.exports import DPORecord, SFTMessage

log = logging.getLogger(__name__)


class DPOPairStrategy(str, enum.Enum):
    FEEDBACK = "feedback"
    FAILURE_RECOVERY = "failure_recovery"
    SYNTHETIC = "synthetic"


def build_dpo_record(
    *,
    prompt: list[SFTMessage],
    chosen: list[SFTMessage],
    rejected: list[SFTMessage],
    metadata: dict[str, Any] | None = None,
) -> DPORecord:
    return DPORecord(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        metadata=metadata or {},
    )


def _from_feedback(traj: Trajectory, *, system_prompt: str | None) -> Iterator[DPORecord]:
    """Read DPO pairs from ``trajectory.tags['feedback_pairs']``.

    Each entry is expected to be:

    .. code-block:: python

        {
          "prompt_event_id": "e_00010",
          "chosen": "Plot A is at 32% moisture.",
          "rejected": "Plot A failed to read.",
          "weight": 1.0,
        }

    The prompt is reconstructed from the trajectory up to (and including) the
    user/tool event referenced by ``prompt_event_id``.
    """
    feedback_pairs = traj.tags.get("feedback_pairs") if isinstance(traj.tags, dict) else None
    if not feedback_pairs:
        return
    base_msgs = trajectory_to_messages(traj, system_prompt=system_prompt)
    # Build an event-id → message-index map, but use string keys for resilience.
    by_eid: dict[str, int] = {}
    cursor = 0
    if system_prompt:
        cursor = 1
    for ev in traj.events:
        eid = ev.event_id
        # Each event consumes 1 message except AssistantEvent that gets folded.
        if isinstance(ev, AssistantEvent):
            continue
        if cursor < len(base_msgs):
            by_eid[eid] = cursor
            cursor += 1

    for fp in feedback_pairs:
        eid = fp.get("prompt_event_id")
        if not eid or eid not in by_eid:
            continue
        end = by_eid[eid] + 1
        prompt = base_msgs[:end]
        chosen = [SFTMessage(role="assistant", content=str(fp.get("chosen") or ""))]
        rejected = [SFTMessage(role="assistant", content=str(fp.get("rejected") or ""))]
        yield build_dpo_record(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={
                "source": "feedback",
                "weight": fp.get("weight", 1.0),
                "session_id": traj.session_id,
                "prompt_event_id": eid,
            },
        )


def _from_failure_recovery(traj: Trajectory, *, system_prompt: str | None) -> Iterator[DPORecord]:
    """Within an agent trace, treat post-recovery assistant text as chosen and
    the pre-recovery (failed) tool-call expression as rejected.

    Concretely we look for the pattern:

        AssistantEvent (text X) | ToolCallEvent T1 → ToolResultEvent (error)
        → ToolCallEvent T2 → ToolResultEvent (ok) → AssistantEvent (text Y)

    and emit a pair with prompt = up to and including the user/tool turn
    immediately before T1, chosen = the recovered Y assistant text, and
    rejected = a pseudo-assistant restating X (or the literal failed call).
    """
    events = traj.events
    base_msgs = trajectory_to_messages(traj, system_prompt=system_prompt)

    for i, ev in enumerate(events):
        if not (isinstance(ev, ToolResultEvent) and ev.is_error):
            continue
        # Look forward for a successful recovery: another ToolCallEvent then
        # a ToolResultEvent that isn't an error and a final AssistantEvent.
        next_assistant: AssistantEvent | None = None
        had_recovery_call = False
        recovery_ok = False
        for follow in events[i + 1 :]:
            if isinstance(follow, ToolCallEvent):
                had_recovery_call = True
            elif isinstance(follow, ToolResultEvent) and had_recovery_call:
                recovery_ok = not follow.is_error
            elif isinstance(follow, AssistantEvent) and recovery_ok:
                next_assistant = follow
                break
        if next_assistant is None:
            continue

        # Walk back to the most recent user/tool result before the failed call,
        # and slice the prompt up to that point.
        end_idx = i  # include the failed result so the model sees the error
        # Convert to message-index by counting events that produce messages
        # (assistant events fold, so we skip them).
        msg_idx = 1 if system_prompt else 0
        for ev2 in events[: end_idx + 1]:
            if isinstance(ev2, AssistantEvent):
                continue
            msg_idx += 1
        prompt = base_msgs[:msg_idx]

        chosen = [SFTMessage(role="assistant", content=next_assistant.content)]
        # Rejected: a generic "stuck" reply — keeps the contrast simple.
        rejected = [
            SFTMessage(
                role="assistant",
                content="I'm sorry, I'm unable to complete the request.",
            )
        ]
        yield build_dpo_record(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={
                "source": "failure_recovery",
                "session_id": traj.session_id,
                "failure_event_id": ev.event_id,
                "recovery_event_id": next_assistant.event_id,
            },
        )


SyntheticGenerator = Callable[
    [Trajectory, list[SFTMessage]], list[tuple[list[SFTMessage], list[SFTMessage]]]
]


def export_dpo_jsonl(
    trajectories: Iterable[Trajectory],
    output_path: str | Path,
    *,
    strategy: DPOPairStrategy | str = DPOPairStrategy.FEEDBACK,
    system_prompt: str | None = None,
    synthetic_generator: SyntheticGenerator | None = None,
) -> int:
    """Build DPO records for each trajectory and write to JSONL."""
    strat = DPOPairStrategy(strategy) if isinstance(strategy, str) else strategy
    rows: Iterator[DPORecord]
    if strat is DPOPairStrategy.FEEDBACK:
        rows = (
            r for traj in trajectories for r in _from_feedback(traj, system_prompt=system_prompt)
        )
    elif strat is DPOPairStrategy.FAILURE_RECOVERY:
        rows = (
            r
            for traj in trajectories
            for r in _from_failure_recovery(traj, system_prompt=system_prompt)
        )
    elif strat is DPOPairStrategy.SYNTHETIC:
        if synthetic_generator is None:
            raise ValueError("synthetic_generator must be provided for the synthetic strategy")

        def _gen() -> Iterator[DPORecord]:
            for traj in trajectories:
                base_msgs = trajectory_to_messages(traj, system_prompt=system_prompt)
                for chosen, rejected in synthetic_generator(traj, base_msgs):
                    yield build_dpo_record(
                        prompt=base_msgs,
                        chosen=chosen,
                        rejected=rejected,
                        metadata={"source": "synthetic", "session_id": traj.session_id},
                    )

        rows = _gen()
    else:  # pragma: no cover
        raise ValueError(f"unknown DPO strategy: {strat}")

    return write_jsonl(output_path, rows)

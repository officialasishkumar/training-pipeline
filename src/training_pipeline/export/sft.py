"""SFT export: trajectory → SFTRecord with chat-template-aligned messages.

The SFT export preserves trajectory structure: tool calls become assistant
messages with ``tool_calls``, tool results become ``role: tool`` messages
referencing the call id. That matches what ``transformers`` chat templates
and TRL ``SFTTrainer`` expect, and is robust to template swaps.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from training_pipeline.ingest.parsers import write_jsonl
from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)
from training_pipeline.schemas.exports import (
    SFTMessage,
    SFTRecord,
    SFTToolCall,
)

log = logging.getLogger(__name__)


def trajectory_to_messages(
    trajectory: Trajectory,
    *,
    system_prompt: str | None = None,
) -> list[SFTMessage]:
    """Render a Trajectory as the ``messages`` list expected by trainers."""
    msgs: list[SFTMessage] = []
    if system_prompt:
        msgs.append(SFTMessage(role="system", content=system_prompt))

    pending_assistant_text: str | None = None
    for ev in trajectory.events:
        if isinstance(ev, UserEvent):
            if pending_assistant_text is not None:
                msgs.append(SFTMessage(role="assistant", content=pending_assistant_text))
                pending_assistant_text = None
            msgs.append(SFTMessage(role="user", content=ev.content))
        elif isinstance(ev, AssistantEvent):
            # Hold the text — it might fold into a following ToolCallEvent at the same turn.
            pending_assistant_text = ev.content if ev.content else None
        elif isinstance(ev, ToolCallEvent):
            calls = [
                SFTToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in ev.tool_calls
            ]
            msgs.append(
                SFTMessage(
                    role="assistant",
                    content=pending_assistant_text,
                    tool_calls=calls,
                )
            )
            pending_assistant_text = None
        elif isinstance(ev, ToolResultEvent):
            if pending_assistant_text is not None:
                msgs.append(SFTMessage(role="assistant", content=pending_assistant_text))
                pending_assistant_text = None
            msgs.append(
                SFTMessage(
                    role="tool",
                    tool_call_id=ev.tool_call_id,
                    name=ev.name,
                    content=ev.content,
                )
            )
    if pending_assistant_text is not None:
        msgs.append(SFTMessage(role="assistant", content=pending_assistant_text))
    return msgs


def build_sft_record(
    trajectory: Trajectory,
    *,
    system_prompt: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SFTRecord:
    """Build a single SFT record from a trajectory."""
    msgs = trajectory_to_messages(trajectory, system_prompt=system_prompt)
    metadata = {
        "session_id": trajectory.session_id,
        "domain": trajectory.domain,
        "source": trajectory.source,
        "schema_version": trajectory.schema_version,
        **trajectory.tags,
        **(extra_metadata or {}),
    }
    return SFTRecord(messages=msgs, metadata=metadata)


def iter_sft_records(
    trajectories: Iterable[Trajectory],
    *,
    system_prompt: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    skip_invalid: bool = True,
) -> Iterator[SFTRecord]:
    for traj in trajectories:
        try:
            yield build_sft_record(traj, system_prompt=system_prompt, extra_metadata=extra_metadata)
        except ValueError as exc:
            if not skip_invalid:
                raise
            log.warning("skipping trajectory %s: %s", traj.session_id, exc)


def export_sft_jsonl(
    trajectories: Iterable[Trajectory],
    output_path: str | Path,
    *,
    system_prompt: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    skip_invalid: bool = True,
) -> int:
    """Write trajectories as SFT JSONL to ``output_path``. Returns row count."""
    return write_jsonl(
        output_path,
        iter_sft_records(
            trajectories,
            system_prompt=system_prompt,
            extra_metadata=extra_metadata,
            skip_invalid=skip_invalid,
        ),
    )

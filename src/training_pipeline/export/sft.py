"""SFT export: trajectory → SFTRecord with chat-template-aligned messages.

The SFT export preserves trajectory structure: tool calls become assistant
messages with ``tool_calls``, tool results become ``role: tool`` messages
referencing the call id. That matches what ``transformers`` chat templates
and TRL ``SFTTrainer`` expect, and is robust to template swaps.

For trainers that support per-token loss weighting, this module also emits
``metadata.loss_weights`` — a parallel array, one float per message, signalling
which messages should contribute to the loss. The default policy weights
assistant turns at 1.0 and zeros the system / user / tool messages so the
model learns to *generate* assistant output without being trained to mimic the
prompt or paste back tool results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
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


LossWeightFn = Callable[[SFTMessage], float]
"""Function: SFTMessage → loss weight (typically 0.0 or 1.0)."""


def default_loss_weight(msg: SFTMessage) -> float:
    """Train on assistant generations only.

    Both pure-text assistant messages and tool-calling assistant messages get
    weight 1.0; the rest of the chat (system/user/tool results) gets 0.0 so
    the model isn't pushed to memorise prompts or repeat observations.
    """
    return 1.0 if msg.role == "assistant" else 0.0


def assistant_text_only_weight(msg: SFTMessage) -> float:
    """Train on natural-language assistant output, *not* tool-call envelopes.

    Useful when the goal is text generation quality and tool selection is
    handled by a separate fine-tune.
    """
    if msg.role != "assistant":
        return 0.0
    return 1.0 if msg.content and not msg.tool_calls else 0.0


_LOSS_POLICIES: dict[str, LossWeightFn] = {
    "assistant_only": default_loss_weight,
    "assistant_text_only": assistant_text_only_weight,
}


def get_loss_policy(name: str) -> LossWeightFn:
    """Resolve a named loss-weight policy."""
    if name not in _LOSS_POLICIES:
        raise KeyError(f"Unknown loss policy {name!r}. Known: {sorted(_LOSS_POLICIES)}")
    return _LOSS_POLICIES[name]


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
    loss_policy: str | LossWeightFn | None = "assistant_only",
) -> SFTRecord:
    """Build a single SFT record from a trajectory.

    ``loss_policy`` controls whether per-message loss weights are emitted:

    - a string name resolved via :func:`get_loss_policy`
    - a callable mapping ``SFTMessage -> float``
    - ``None`` to skip emission entirely (kept for backward compat)
    """
    msgs = trajectory_to_messages(trajectory, system_prompt=system_prompt)
    metadata: dict[str, Any] = {
        "session_id": trajectory.session_id,
        "domain": trajectory.domain,
        "source": trajectory.source,
        "schema_version": trajectory.schema_version,
        **trajectory.tags,
        **(extra_metadata or {}),
    }
    if trajectory.lineage_id is not None:
        metadata["lineage_id"] = trajectory.lineage_id
    if loss_policy is not None:
        fn = get_loss_policy(loss_policy) if isinstance(loss_policy, str) else loss_policy
        metadata["loss_weights"] = [fn(m) for m in msgs]
        metadata["loss_policy"] = loss_policy if isinstance(loss_policy, str) else "custom"
    return SFTRecord(messages=msgs, metadata=metadata)


def iter_sft_records(
    trajectories: Iterable[Trajectory],
    *,
    system_prompt: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    skip_invalid: bool = True,
    loss_policy: str | LossWeightFn | None = "assistant_only",
) -> Iterator[SFTRecord]:
    for traj in trajectories:
        try:
            yield build_sft_record(
                traj,
                system_prompt=system_prompt,
                extra_metadata=extra_metadata,
                loss_policy=loss_policy,
            )
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
    loss_policy: str | LossWeightFn | None = "assistant_only",
) -> int:
    """Write trajectories as SFT JSONL to ``output_path``. Returns row count."""
    return write_jsonl(
        output_path,
        iter_sft_records(
            trajectories,
            system_prompt=system_prompt,
            extra_metadata=extra_metadata,
            skip_invalid=skip_invalid,
            loss_policy=loss_policy,
        ),
    )

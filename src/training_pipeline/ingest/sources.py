"""Source-format adapters.

Each adapter takes a single raw log record (a dict, as parsed from JSONL) and
maps it onto a canonical ``Trajectory``. Adapters are registered by name so
the CLI can be configured without code changes.

Built-in adapters:

- ``openai_chat``     — OpenAI Chat Completions API logs (with tool_calls)
- ``anthropic``       — Anthropic Messages API logs (with tool_use blocks)
- ``generic_chat``    — generic ``messages: [{role, content}, ...]`` logs
- ``canonical``       — already in this pipeline's canonical schema

Every adapter assigns a ``lineage_id`` derived from the raw record content. The
id survives every later stage so an exported row can be traced back to the
exact log line it came from — useful for audits, replay, and post-hoc
debugging when a downstream review surfaces an issue with a specific row.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

import orjson

from training_pipeline.schemas.events import (
    AssistantEvent,
    Event,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)

SourceAdapter = Callable[[dict[str, Any]], Trajectory]

_REGISTRY: dict[str, SourceAdapter] = {}


def register_source(name: str) -> Callable[[SourceAdapter], SourceAdapter]:
    """Decorator: register an adapter under ``name``."""

    def deco(fn: SourceAdapter) -> SourceAdapter:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_source(name: str) -> SourceAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown source adapter: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def detect_source(record: dict[str, Any]) -> str:
    """Best-effort sniff of the source format for a single record.

    Returns the adapter name. ``generic_chat`` is the catch-all.
    """
    if "events" in record and "session_id" in record and isinstance(record.get("events"), list):
        return "canonical"
    if "messages" in record:
        msgs = record["messages"]
        if isinstance(msgs, list) and msgs:
            first = msgs[0]
            if isinstance(first, dict):
                # Anthropic uses content blocks (a list); OpenAI uses string content.
                if isinstance(first.get("content"), list) and any(
                    isinstance(b, dict) and b.get("type") in ("text", "tool_use", "tool_result")
                    for b in first["content"]
                ):
                    return "anthropic"
                if any(
                    isinstance(m, dict)
                    and m.get("role") in ("tool", "assistant")
                    and (m.get("tool_calls") is not None or m.get("tool_call_id") is not None)
                    for m in msgs
                ):
                    return "openai_chat"
        return "generic_chat"
    return "generic_chat"


def _ts(record: dict[str, Any], default: datetime | None = None) -> datetime:
    val = record.get("timestamp") or record.get("created_at") or record.get("time")
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(float(val), tz=timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return default or datetime.now(tz=timezone.utc)


def _sid(record: dict[str, Any]) -> str:
    return str(
        record.get("session_id")
        or record.get("conversation_id")
        or record.get("id")
        or uuid.uuid4().hex
    )


def _lineage_id(record: dict[str, Any]) -> str:
    """Stable id for this raw record.

    Prefer an explicit ``lineage_id`` on the record (so re-ingests produce the
    same id), then fall back to a content hash so identical inputs always
    produce identical lineage ids — important for reproducibility.
    """
    explicit = record.get("lineage_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    try:
        payload = orjson.dumps(record, option=orjson.OPT_SORT_KEYS)
    except TypeError:
        payload = repr(record).encode("utf-8")
    return "lin_" + hashlib.sha256(payload).hexdigest()[:16]


def _attach_lineage(events: list[Event], lineage_id: str) -> list[Event]:
    """Stamp ``lineage_id`` on every event in-place via model_copy."""
    return [ev.model_copy(update={"lineage_id": lineage_id}) for ev in events]


def _next_id(prefix: str, n: int) -> str:
    return f"{prefix}_{n:05d}"


@register_source("canonical")
def from_canonical(record: dict[str, Any]) -> Trajectory:
    """Pass-through: already-canonical records just need re-validation.

    Lineage is preserved if present; otherwise a fresh id is derived from the
    record content so re-ingesting the same JSON always yields the same id.
    """
    lineage = _lineage_id(record)
    traj = Trajectory.model_validate(record)
    if traj.lineage_id is None:
        traj = traj.model_copy(
            update={
                "lineage_id": lineage,
                "events": _attach_lineage(traj.events, lineage),
            }
        )
    return traj


@register_source("generic_chat")
def from_generic_chat(record: dict[str, Any]) -> Trajectory:
    """Adapt a generic ``messages: [{role, content}]`` record."""
    sid = _sid(record)
    base_ts = _ts(record)
    events: list[Event] = []
    for i, msg in enumerate(record.get("messages", [])):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        ts = _ts(msg, default=base_ts)
        evt_id = _next_id("e", i)
        if role == "user":
            events.append(
                UserEvent(
                    event_id=evt_id,
                    session_id=sid,
                    timestamp=ts,
                    content=str(msg.get("content", "")),
                )
            )
        elif role == "assistant":
            events.append(
                AssistantEvent(
                    event_id=evt_id,
                    session_id=sid,
                    timestamp=ts,
                    content=str(msg.get("content", "")),
                )
            )
    lineage = _lineage_id(record)
    return Trajectory(
        session_id=sid,
        events=_attach_lineage(events, lineage),
        source=record.get("source", "generic_chat"),
        domain=record.get("domain"),
        tags=dict(record.get("tags") or {}),
        lineage_id=lineage,
    )


@register_source("openai_chat")
def from_openai_chat(record: dict[str, Any]) -> Trajectory:
    """Adapt an OpenAI Chat Completions record (with tool_calls and tool messages)."""
    sid = _sid(record)
    base_ts = _ts(record)
    events: list[Event] = []
    for i, msg in enumerate(record.get("messages", [])):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        ts = _ts(msg, default=base_ts)
        evt_id = _next_id("e", i)
        content = msg.get("content")
        # OpenAI sometimes uses content as a list of parts (vision/multimodal).
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content_str = "" if content is None else str(content)
        if role == "user":
            events.append(
                UserEvent(event_id=evt_id, session_id=sid, timestamp=ts, content=content_str)
            )
        elif role == "assistant":
            tool_calls_raw = msg.get("tool_calls") or []
            if tool_calls_raw:
                # Emit assistant text first if present.
                if content_str:
                    events.append(
                        AssistantEvent(
                            event_id=evt_id + "t",
                            session_id=sid,
                            timestamp=ts,
                            content=content_str,
                        )
                    )
                calls = [_openai_tool_call(tc) for tc in tool_calls_raw]
                events.append(
                    ToolCallEvent(event_id=evt_id, session_id=sid, timestamp=ts, tool_calls=calls)
                )
            else:
                events.append(
                    AssistantEvent(
                        event_id=evt_id,
                        session_id=sid,
                        timestamp=ts,
                        content=content_str,
                        finish_reason=msg.get("finish_reason"),
                    )
                )
        elif role == "tool":
            events.append(
                ToolResultEvent(
                    event_id=evt_id,
                    session_id=sid,
                    timestamp=ts,
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    name=str(msg.get("name") or "unknown"),
                    content=content_str,
                    is_error=bool(msg.get("is_error", False)),
                )
            )
    lineage = _lineage_id(record)
    return Trajectory(
        session_id=sid,
        events=_attach_lineage(events, lineage),
        source=record.get("source", "openai_chat"),
        domain=record.get("domain"),
        tags=dict(record.get("tags") or {}),
        lineage_id=lineage,
    )


def _openai_tool_call(tc: dict[str, Any]) -> ToolCall:
    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
    args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
    if isinstance(args, str):
        try:
            import orjson

            args = orjson.loads(args)
        except Exception:
            args = {"_raw": args}
    return ToolCall(
        id=str(tc.get("id") or uuid.uuid4().hex),
        name=str(fn.get("name") or tc.get("name") or "unknown"),
        arguments=args if isinstance(args, dict) else {"_raw": args},
    )


@register_source("anthropic")
def from_anthropic(record: dict[str, Any]) -> Trajectory:
    """Adapt an Anthropic Messages record where content may be a list of blocks."""
    sid = _sid(record)
    base_ts = _ts(record)
    events: list[Event] = []
    counter = 0
    for msg in record.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        ts = _ts(msg, default=base_ts)
        content = msg.get("content")
        if isinstance(content, str):
            content_blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_blocks = [b for b in content if isinstance(b, dict)]
        else:
            content_blocks = []

        if role == "user":
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            tool_results = [b for b in content_blocks if b.get("type") == "tool_result"]
            counter += 1
            if text:
                events.append(
                    UserEvent(
                        event_id=_next_id("e", counter),
                        session_id=sid,
                        timestamp=ts,
                        content=text,
                    )
                )
            for tr in tool_results:
                counter += 1
                tr_content = tr.get("content")
                if isinstance(tr_content, list):
                    tr_content = "".join(
                        b.get("text", "") for b in tr_content if isinstance(b, dict)
                    )
                events.append(
                    ToolResultEvent(
                        event_id=_next_id("e", counter),
                        session_id=sid,
                        timestamp=ts,
                        tool_call_id=str(tr.get("tool_use_id") or ""),
                        name=str(tr.get("name") or "unknown"),
                        content=str(tr_content or ""),
                        is_error=bool(tr.get("is_error", False)),
                    )
                )
        elif role == "assistant":
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
            counter += 1
            if text:
                events.append(
                    AssistantEvent(
                        event_id=_next_id("e", counter),
                        session_id=sid,
                        timestamp=ts,
                        content=text,
                    )
                )
            if tool_uses:
                counter += 1
                calls = [
                    ToolCall(
                        id=str(b.get("id") or uuid.uuid4().hex),
                        name=str(b.get("name") or "unknown"),
                        arguments=b.get("input") or {},
                    )
                    for b in tool_uses
                ]
                events.append(
                    ToolCallEvent(
                        event_id=_next_id("e", counter),
                        session_id=sid,
                        timestamp=ts,
                        tool_calls=calls,
                    )
                )
    lineage = _lineage_id(record)
    return Trajectory(
        session_id=sid,
        events=_attach_lineage(events, lineage),
        source=record.get("source", "anthropic"),
        domain=record.get("domain"),
        tags=dict(record.get("tags") or {}),
        lineage_id=lineage,
    )


def known_sources() -> Iterable[str]:
    """Names of registered adapters."""
    return tuple(sorted(_REGISTRY))

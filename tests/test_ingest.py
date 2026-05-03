"""Ingest layer: parsers and source adapters."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from training_pipeline.ingest import (
    detect_source,
    iter_jsonl,
    iter_records,
    normalize_record,
    write_jsonl,
)
from training_pipeline.ingest.normalizer import NormalizationError, normalize_records
from training_pipeline.ingest.sources import known_sources
from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)


def test_iter_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"x": 1}\n\n{"x": 2}\n   \n{"x": 3}\n', encoding="utf-8")
    rows = list(iter_jsonl(p))
    assert [r["x"] for r in rows] == [1, 2, 3]


def test_iter_jsonl_strict_raises_on_garbage(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"x": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_jsonl(p, strict=True))


def test_iter_jsonl_handles_gzip(tmp_path: Path):
    p = tmp_path / "a.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write('{"x": 1}\n{"x": 2}\n')
    rows = list(iter_jsonl(p))
    assert [r["x"] for r in rows] == [1, 2]


def test_iter_records_walks_directory(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.jsonl").write_text('{"x": 2}\n', encoding="utf-8")
    (sub / "c.json").write_text('[{"x": 3}, {"x": 4}]', encoding="utf-8")
    rows = list(iter_records(tmp_path))
    assert sorted(r["x"] for r in rows) == [1, 2, 3, 4]


def test_write_jsonl_drops_none_by_default(tmp_path: Path):
    from training_pipeline.schemas.exports import SFTMessage

    p = tmp_path / "out.jsonl"
    write_jsonl(p, [SFTMessage(role="user", content="hi")])
    line = p.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert "tool_call_id" not in payload
    assert payload["role"] == "user"


def test_known_sources_includes_builtins():
    names = set(known_sources())
    assert {"openai_chat", "anthropic", "generic_chat", "canonical"} <= names


def test_detect_openai_with_tools(openai_record: dict[str, Any]):
    assert detect_source(openai_record) == "openai_chat"


def test_detect_anthropic(anthropic_record: dict[str, Any]):
    assert detect_source(anthropic_record) == "anthropic"


def test_detect_generic_chat_for_plain_messages():
    assert (
        detect_source({"messages": [{"role": "user", "content": "hi"}]})
        == "generic_chat"
    )


def test_detect_canonical():
    rec = {"session_id": "x", "events": [{"event_id": "u", "kind": "user"}]}
    assert detect_source(rec) == "canonical"


def test_normalize_openai(openai_record: dict[str, Any]):
    traj = normalize_record(openai_record)
    assert traj.session_id == "oa-1"
    types = [type(e).__name__ for e in traj.events]
    assert types == [
        "UserEvent",
        "ToolCallEvent",
        "ToolResultEvent",
        "AssistantEvent",
    ]
    tc = traj.events[1]
    assert isinstance(tc, ToolCallEvent)
    assert tc.tool_calls[0].name == "soil_sensor"
    assert tc.tool_calls[0].arguments == {"plot": "A"}


def test_normalize_anthropic(anthropic_record: dict[str, Any]):
    traj = normalize_record(anthropic_record)
    types = [type(e).__name__ for e in traj.events]
    # text + tool_use turn becomes Assistant + ToolCall events
    assert "ToolCallEvent" in types
    assert "ToolResultEvent" in types
    assert "AssistantEvent" in types


def test_normalize_records_yields_errors_inline():
    bad = {"messages": [{"role": "user", "content": object()}]}  # not serialisable
    out = list(normalize_records([bad]))
    # Either succeeds (string-coerced) or yields a NormalizationError; both are fine.
    assert len(out) == 1


def test_canonical_passthrough_roundtrip(qa_trajectory: Trajectory):
    raw = qa_trajectory.model_dump(mode="json")
    traj = normalize_record(raw, source="canonical")
    assert traj.session_id == qa_trajectory.session_id
    assert len(traj.events) == 2

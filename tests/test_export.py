"""Export: SFT, DPO, templates, sharding."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from training_pipeline.export.dpo import DPOPairStrategy, export_dpo_jsonl
from training_pipeline.export.sft import (
    build_sft_record,
    export_sft_jsonl,
    trajectory_to_messages,
)
from training_pipeline.export.shards import ShardWriter, write_dataset_card
from training_pipeline.export.templates import (
    KNOWN_TEMPLATES,
    apply_template,
    template_for,
)
from training_pipeline.schemas.events import Trajectory


def test_qa_trajectory_to_messages(qa_trajectory: Trajectory):
    msgs = trajectory_to_messages(qa_trajectory, system_prompt="You are nice.")
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert msgs[2].role == "assistant"


def test_agent_trajectory_to_messages(agent_trajectory: Trajectory):
    msgs = trajectory_to_messages(agent_trajectory)
    roles = [m.role for m in msgs]
    # user → tool_call_assistant → tool → tool_call_assistant → tool → final assistant
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    # Tool calls preserved.
    tc_msgs = [m for m in msgs if m.role == "assistant" and m.tool_calls]
    assert len(tc_msgs) == 2


def test_build_sft_record_carries_metadata(qa_trajectory: Trajectory):
    rec = build_sft_record(qa_trajectory, extra_metadata={"split": "train"})
    assert rec.metadata["session_id"] == "qa-1"
    assert rec.metadata["domain"] == "agronomy"
    assert rec.metadata["split"] == "train"


def test_export_sft_jsonl_file_count(qa_trajectory: Trajectory, tmp_path: Path):
    p = tmp_path / "out.jsonl"
    n = export_sft_jsonl([qa_trajectory], p)
    assert n == 1
    payload = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert "messages" in payload


def test_template_chatml_renders(qa_trajectory: Trajectory):
    msgs = trajectory_to_messages(qa_trajectory)
    rendered = apply_template(msgs, template="chatml")
    assert "<|im_start|>user" in rendered
    assert "<|im_end|>" in rendered


def test_template_for_unknown_raises():
    with pytest.raises(KeyError):
        template_for("definitely_not_a_template")


@pytest.mark.parametrize("template_name", list(KNOWN_TEMPLATES))
def test_every_known_template_renders(template_name, qa_trajectory: Trajectory):
    msgs = trajectory_to_messages(qa_trajectory)
    rendered = apply_template(msgs, template=template_name)
    assert isinstance(rendered, str) and rendered


def test_dpo_feedback_strategy(tmp_path: Path):
    """DPO feedback pairs are read from trajectory tags."""
    from datetime import datetime, timezone

    from training_pipeline.schemas.events import (
        AssistantEvent,
        Trajectory,
        UserEvent,
    )

    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(
                event_id="u1",
                session_id="s",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                content="What's the answer?",
            ),
            AssistantEvent(
                event_id="a1",
                session_id="s",
                timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                content="It is 42.",
            ),
        ],
        tags={
            "feedback_pairs": [
                {
                    "prompt_event_id": "u1",
                    "chosen": "It is 42.",
                    "rejected": "I don't know.",
                }
            ]
        },
    )
    p = tmp_path / "dpo.jsonl"
    n = export_dpo_jsonl([traj], p, strategy=DPOPairStrategy.FEEDBACK)
    assert n == 1
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["chosen"][0]["content"] == "It is 42."
    assert payload["rejected"][0]["content"] == "I don't know."


def test_dpo_failure_recovery_strategy(agent_trajectory: Trajectory, tmp_path: Path):
    p = tmp_path / "dpo.jsonl"
    n = export_dpo_jsonl([agent_trajectory], p, strategy=DPOPairStrategy.FAILURE_RECOVERY)
    assert n >= 1
    rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert rec["chosen"]
    assert rec["rejected"]
    assert rec["metadata"]["source"] == "failure_recovery"


def test_shard_writer_rotates(tmp_path: Path):
    with ShardWriter(tmp_path, shard_size=2, prefix="x") as w:
        for i in range(5):
            w.write({"i": i})
    files = sorted(p.name for p in tmp_path.glob("x-*.jsonl"))
    assert len(files) == 3  # 2, 2, 1


def test_shard_writer_compress(tmp_path: Path):
    with ShardWriter(tmp_path, shard_size=10, prefix="x", compress=True) as w:
        w.write({"hello": "world"})
    files = list(tmp_path.glob("x-*.jsonl.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt") as f:
        assert json.loads(f.read().strip())["hello"] == "world"


def test_dataset_card_written(tmp_path: Path):
    write_dataset_card(
        tmp_path,
        name="sft",
        record_count=10,
        fingerprint="deadbeef",
        fields=["messages"],
        chat_template="chatml",
    )
    card = json.loads((tmp_path / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["record_count"] == 10
    assert card["chat_template"] == "chatml"
    assert "training_pipeline_version" in card

"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from training_pipeline.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_logs(tmp_path: Path) -> Path:
    """A directory containing one OpenAI-style log line."""
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps(
            {
                "session_id": "s",
                "domain": "agronomy",
                "messages": [
                    {"role": "user", "content": "Email user@example.com"},
                    {
                        "role": "assistant",
                        "content": "I'll keep that in mind.",
                    },
                ],
            }
        )
        + "\n"
    )
    return p


def test_version(runner: CliRunner):
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "training-pipeline" in result.stdout


def test_ingest_writes_canonical(runner: CliRunner, sample_logs: Path, tmp_path: Path):
    out = tmp_path / "canon.jsonl"
    result = runner.invoke(app, ["ingest", "--input", str(sample_logs), "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["session_id"] == "s"


def test_redact_replaces_email(runner: CliRunner, sample_logs: Path, tmp_path: Path):
    canon = tmp_path / "canon.jsonl"
    runner.invoke(app, ["ingest", "--input", str(sample_logs), "--output", str(canon)])
    out = tmp_path / "redacted.jsonl"
    result = runner.invoke(app, ["redact", "--input", str(canon), "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    text = out.read_text(encoding="utf-8")
    assert "@example.com" not in text


def test_full_chain_via_cli(runner: CliRunner, sample_logs: Path, tmp_path: Path):
    canon = tmp_path / "canon.jsonl"
    redacted = tmp_path / "red.jsonl"
    tagged = tmp_path / "tagged.jsonl"
    sft_dir = tmp_path / "sft"
    runner.invoke(app, ["ingest", "-i", str(sample_logs), "-o", str(canon)])
    runner.invoke(app, ["redact", "-i", str(canon), "-o", str(redacted)])
    runner.invoke(app, ["tag", "-i", str(redacted), "-o", str(tagged)])
    result = runner.invoke(
        app,
        [
            "export",
            "sft",
            "-i",
            str(tagged),
            "-o",
            str(sft_dir),
            "--shard-size",
            "100",
        ],
    )
    assert result.exit_code == 0, result.stdout
    shards = list(sft_dir.glob("sft-*.jsonl"))
    assert shards
    rec = json.loads(shards[0].read_text(encoding="utf-8").splitlines()[0])
    assert rec["messages"][0]["role"] in ("user", "system")

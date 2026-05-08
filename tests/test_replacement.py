"""Replacement-criteria rubric.

End-to-end coverage on the structured-outputs path (no GPU). The
production runner is exercised by an isolation test that just checks
the import paths exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from training_pipeline.cli import app
from training_pipeline.eval.replacement import (
    EDGE_CASE_CATEGORIES,
    ReplacementThresholds,
    StaticOutputsRunner,
    _percentile,
    evaluate_replacement,
)


def _suite_row(
    *,
    pid: str,
    category: str,
    gold_calls: list[dict] | None = None,
    success: float = 1.0,
) -> dict:
    return {
        "prompt_id": pid,
        "edge_case_category": category,
        "prompt": [{"role": "user", "content": "x"}],
        "gold_tool_calls": gold_calls or [],
        "gold_success": success,
    }


def _output_row(
    *,
    pid: str,
    tool_calls: list[dict] | None = None,
    persona: float = 1.0,
    latency: float = 1500.0,
    success: float = 1.0,
) -> dict:
    return {
        "prompt_id": pid,
        "tool_calls": tool_calls or [],
        "text": "ok",
        "persona_score": persona,
        "latency_ms": latency,
        "success": success,
    }


@pytest.fixture
def suite_with_two_categories(tmp_path: Path) -> Path:
    rows = [
        _suite_row(pid="p1", category="pure_qa"),
        _suite_row(pid="p2", category="pure_qa"),
        _suite_row(pid="p3", category="multi_tool"),
        _suite_row(pid="p4", category="multi_tool"),
    ]
    p = tmp_path / "suite.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_percentile_handles_short_lists():
    assert _percentile([], 0.95) == 0.0
    assert _percentile([10.0], 0.95) == 10.0
    assert _percentile([10.0, 20.0], 0.5) == 15.0


def test_static_outputs_runner_fetches_by_prompt_id(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"prompt_id": "p1", "text": "hello", "latency_ms": 200.0}) + "\n"
    )
    r = StaticOutputsRunner.from_jsonl("teacher", p)
    assert r.fetch("p1")["text"] == "hello"
    assert r.fetch("missing") == {"tool_calls": [], "text": "", "latency_ms": 0.0}


def test_evaluate_replacement_accepts_when_student_matches(
    tmp_path: Path, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, persona=0.9, latency=2000.0))
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    student = tmp_path / "s.jsonl"
    # Student matches persona, latency strictly better.
    student.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, persona=0.9, latency=1500.0))
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    verdict = evaluate_replacement(
        teacher_outputs_path=teacher,
        student_outputs_path=student,
        eval_set_path=suite_with_two_categories,
        teacher_params=32_000_000_000,
        student_params=8_000_000_000,
        student_context_window=32_768,
    )
    assert verdict.accepted, verdict.reasons
    assert {c.category for c in verdict.by_category} == {"pure_qa", "multi_tool"}


def test_evaluate_replacement_rejects_quality_regression(
    tmp_path: Path, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, persona=0.95)) for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    student = tmp_path / "s.jsonl"
    # Student persona drops below 0.95x in pure_qa only.
    student.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, persona=0.6 if p in ("p1", "p2") else 0.95))
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    verdict = evaluate_replacement(
        teacher_outputs_path=teacher,
        student_outputs_path=student,
        eval_set_path=suite_with_two_categories,
    )
    assert not verdict.accepted
    assert any("pure_qa/persona_adherence_rate" in r for r in verdict.reasons)


def test_evaluate_replacement_rejects_latency_regression(
    tmp_path: Path, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, latency=2000.0))
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    student = tmp_path / "s.jsonl"
    student.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, latency=2200.0))  # worse than teacher
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    verdict = evaluate_replacement(
        teacher_outputs_path=teacher,
        student_outputs_path=student,
        eval_set_path=suite_with_two_categories,
    )
    assert not verdict.accepted
    assert any("latency_p95_ms" in r for r in verdict.reasons)


def test_evaluate_replacement_rejects_when_student_larger(
    tmp_path: Path, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(json.dumps(_output_row(pid=p)) for p in ("p1", "p2", "p3", "p4")) + "\n"
    )
    student = tmp_path / "s.jsonl"
    student.write_text(
        "\n".join(json.dumps(_output_row(pid=p)) for p in ("p1", "p2", "p3", "p4")) + "\n"
    )
    verdict = evaluate_replacement(
        teacher_outputs_path=teacher,
        student_outputs_path=student,
        eval_set_path=suite_with_two_categories,
        teacher_params=8_000_000_000,
        student_params=32_000_000_000,  # bigger
    )
    assert not verdict.accepted
    assert any("params" in r for r in verdict.reasons)


def test_evaluate_replacement_rejects_when_context_too_small(
    tmp_path: Path, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(json.dumps(_output_row(pid=p)) for p in ("p1", "p2", "p3", "p4")) + "\n"
    )
    student = tmp_path / "s.jsonl"
    student.write_text(
        "\n".join(json.dumps(_output_row(pid=p)) for p in ("p1", "p2", "p3", "p4")) + "\n"
    )
    verdict = evaluate_replacement(
        teacher_outputs_path=teacher,
        student_outputs_path=student,
        eval_set_path=suite_with_two_categories,
        student_context_window=8_000,
    )
    assert not verdict.accepted
    assert any("context_window" in r for r in verdict.reasons)


def test_edge_case_categories_match_doc():
    assert "pure_qa" in EDGE_CASE_CATEGORIES
    assert "tool_failure_recovery" in EDGE_CASE_CATEGORIES
    assert "multilingual" in EDGE_CASE_CATEGORIES
    assert "jailbreak_refusal" in EDGE_CASE_CATEGORIES


def test_thresholds_carry_defaults():
    th = ReplacementThresholds()
    assert th.quality_ratio_floor == 0.95
    assert th.latency_p95_target_ms == 4000.0
    assert th.context_window_min == 32_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_eval_compare_accepts(
    tmp_path: Path, cli_runner: CliRunner, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, persona=0.9, latency=2200.0))
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    student = tmp_path / "s.jsonl"
    student.write_text(
        "\n".join(
            json.dumps(_output_row(pid=p, persona=0.9, latency=1800.0))
            for p in ("p1", "p2", "p3", "p4")
        )
        + "\n"
    )
    result = cli_runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--teacher",
            str(teacher),
            "--student",
            str(student),
            "--suite",
            str(suite_with_two_categories),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "ACCEPTED" in result.stdout


def test_cli_eval_compare_rejects_with_nonzero_exit(
    tmp_path: Path, cli_runner: CliRunner, suite_with_two_categories: Path
):
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        "\n".join(json.dumps(_output_row(pid=p, persona=0.95)) for p in ("p1", "p2", "p3", "p4"))
        + "\n"
    )
    student = tmp_path / "s.jsonl"
    student.write_text(
        "\n".join(json.dumps(_output_row(pid=p, persona=0.5)) for p in ("p1", "p2", "p3", "p4"))
        + "\n"
    )
    result = cli_runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--teacher",
            str(teacher),
            "--student",
            str(student),
            "--suite",
            str(suite_with_two_categories),
        ],
    )
    assert result.exit_code == 2
    assert "REJECTED" in result.stdout


def test_cli_eval_outputs_legacy_still_works(
    tmp_path: Path, cli_runner: CliRunner
):
    """Legacy `tp eval outputs` path still works (renamed from `tp eval`)."""
    teacher = tmp_path / "t.jsonl"
    teacher.write_text(
        json.dumps(
            {
                "prompt_id": "p",
                "tool_calls": [{"name": "x", "arguments": {}}],
                "text": "ok",
            }
        )
        + "\n"
    )
    student = tmp_path / "s.jsonl"
    student.write_text(
        json.dumps(
            {
                "prompt_id": "p",
                "tool_calls": [{"name": "x", "arguments": {}}],
                "text": "ok",
            }
        )
        + "\n"
    )
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps(
            {
                "prompt_id": "p",
                "prompt": [{"role": "user", "content": "x"}],
                "gold_tool_calls": [{"name": "x", "arguments": {}}],
                "gold_text": "ok",
            }
        )
        + "\n"
    )
    result = cli_runner.invoke(
        app,
        [
            "eval",
            "outputs",
            "--teacher",
            str(teacher),
            "--student",
            str(student),
            "--eval-set",
            str(eval_set),
        ],
    )
    assert result.exit_code == 0, result.stdout

"""Eval comparator and tool-use scoring."""

from __future__ import annotations

import json
from pathlib import Path

from training_pipeline.eval.compare import compare_outputs
from training_pipeline.eval.tool_use import score_tool_use
from training_pipeline.validate.consistency import ToolRegistry, ToolSpec


def test_score_tool_use_perfect_match():
    pairs = [
        (
            [{"name": "x", "arguments": {"a": 1}}],
            [{"name": "x", "arguments": {"a": 1}}],
        ),
    ]
    s = score_tool_use(pairs)
    assert s.tool_name_accuracy == 1.0
    assert s.arg_exact_match == 1.0


def test_score_tool_use_wrong_args_penalty():
    pairs = [
        (
            [{"name": "x", "arguments": {"a": 1}}],
            [{"name": "x", "arguments": {"a": 2}}],
        ),
    ]
    s = score_tool_use(pairs)
    assert s.tool_name_accuracy == 1.0
    assert s.arg_exact_match == 0.0


def test_score_tool_use_schema_validity():
    registry = ToolRegistry(
        tools={"x": ToolSpec(name="x", required=("a",), properties={"a": "number"})}
    )
    pairs = [
        (
            [{"name": "x", "arguments": {"a": "not a number"}}],
            [{"name": "x", "arguments": {"a": 1}}],
        ),
    ]
    s = score_tool_use(pairs, registry=registry)
    assert s.schema_validity == 0.0


def test_compare_outputs_basic(tmp_path: Path):
    eval_set = [
        {
            "prompt_id": "p1",
            "prompt": [{"role": "user", "content": "x"}],
            "gold_tool_calls": [{"name": "x", "arguments": {"a": 1}}],
            "gold_text": "the answer is 42",
        }
    ]
    student = [
        {
            "prompt_id": "p1",
            "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
            "text": "the answer is 42",
        }
    ]
    teacher = [
        {
            "prompt_id": "p1",
            "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
            "text": "the answer is 42",
        }
    ]
    eval_path = tmp_path / "eval.jsonl"
    student_path = tmp_path / "student.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_set))
    student_path.write_text("\n".join(json.dumps(r) for r in student))
    teacher_path.write_text("\n".join(json.dumps(r) for r in teacher))
    summary = compare_outputs(
        student=student_path, teacher=teacher_path, eval_set=eval_path
    )
    assert summary["n_prompts"] == 1
    assert summary["metrics"]["tool_name_accuracy"]["student"] == 1.0
    assert summary["summary"]["student_acceptable"] is True


def test_compare_outputs_flags_regression(tmp_path: Path):
    eval_set = [
        {
            "prompt_id": f"p{i}",
            "prompt": [{"role": "user", "content": "x"}],
            "gold_tool_calls": [{"name": "x", "arguments": {"a": 1}}],
            "gold_text": "ok",
        }
        for i in range(10)
    ]
    student = [
        {"prompt_id": f"p{i}", "tool_calls": [], "text": "wrong"} for i in range(10)
    ]
    teacher = [
        {
            "prompt_id": f"p{i}",
            "tool_calls": [{"name": "x", "arguments": {"a": 1}}],
            "text": "ok",
        }
        for i in range(10)
    ]
    eval_path = tmp_path / "eval.jsonl"
    student_path = tmp_path / "student.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_set))
    student_path.write_text("\n".join(json.dumps(r) for r in student))
    teacher_path.write_text("\n".join(json.dumps(r) for r in teacher))
    summary = compare_outputs(
        student=student_path, teacher=teacher_path, eval_set=eval_path
    )
    assert summary["summary"]["student_acceptable"] is False
    assert "tool_name_accuracy" in summary["summary"]["regressed_metrics"]

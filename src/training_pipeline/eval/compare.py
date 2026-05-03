"""Compare teacher and student outputs against an eval set.

Eval-set JSONL schema (one line per prompt):

.. code-block:: json

    {
      "prompt_id": "p_001",
      "prompt": [{"role": "system", "..."}, {"role": "user", "..."}],
      "gold_tool_calls": [{"name": "soil_sensor", "arguments": {"plot": "A"}}],
      "gold_text": "Plot A is at 32% moisture."
    }

Outputs JSONL (per model):

.. code-block:: json

    {"prompt_id": "p_001", "tool_calls": [...], "text": "..."}

This module computes per-model metrics and a summary delta. It does not run a
model — that's a trainer / inference-server concern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from training_pipeline.eval.tool_use import score_tool_use
from training_pipeline.ingest.parsers import iter_records

log = logging.getLogger(__name__)


def _load_outputs(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r.get("prompt_id")): r for r in iter_records(path) if r.get("prompt_id")}


def _text_overlap(a: str, b: str) -> float:
    """Cheap unigram F1; not a substitute for proper eval but useful as a smoke check."""
    aw = a.lower().split()
    bw = b.lower().split()
    if not aw or not bw:
        return 0.0
    common = set(aw) & set(bw)
    if not common:
        return 0.0
    p = len(common) / len(set(aw))
    r = len(common) / len(set(bw))
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compare_outputs(
    *,
    student: Path | str,
    teacher: Path | str,
    eval_set: Path | str,
) -> dict[str, Any]:
    """Compute metrics for student/teacher against the eval set.

    Returns a structured summary:

    .. code-block:: json

        {
          "n_prompts": 100,
          "metrics": {
            "tool_name_accuracy": {"teacher": 0.91, "student": 0.85, "delta": -0.06},
            "arg_exact_match":   {"teacher": 0.78, "student": 0.71, "delta": -0.07},
            "text_f1":           {"teacher": 0.62, "student": 0.58, "delta": -0.04}
          }
        }
    """
    student_out = _load_outputs(Path(student))
    teacher_out = _load_outputs(Path(teacher))
    eval_records = list(iter_records(Path(eval_set)))

    student_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    teacher_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    student_text_f1: list[float] = []
    teacher_text_f1: list[float] = []

    for rec in eval_records:
        pid = str(rec.get("prompt_id"))
        gold_calls = rec.get("gold_tool_calls", []) or []
        gold_text = rec.get("gold_text", "")
        s = student_out.get(pid)
        t = teacher_out.get(pid)
        if s is not None:
            student_pairs.append((s.get("tool_calls", []) or [], gold_calls))
            student_text_f1.append(_text_overlap(s.get("text", ""), gold_text))
        if t is not None:
            teacher_pairs.append((t.get("tool_calls", []) or [], gold_calls))
            teacher_text_f1.append(_text_overlap(t.get("text", ""), gold_text))

    s_score = score_tool_use(student_pairs)
    t_score = score_tool_use(teacher_pairs)
    s_text = sum(student_text_f1) / len(student_text_f1) if student_text_f1 else 0.0
    t_text = sum(teacher_text_f1) / len(teacher_text_f1) if teacher_text_f1 else 0.0

    metrics: dict[str, dict[str, float]] = {
        "tool_name_accuracy": {
            "teacher": t_score.tool_name_accuracy,
            "student": s_score.tool_name_accuracy,
            "delta": s_score.tool_name_accuracy - t_score.tool_name_accuracy,
        },
        "arg_exact_match": {
            "teacher": t_score.arg_exact_match,
            "student": s_score.arg_exact_match,
            "delta": s_score.arg_exact_match - t_score.arg_exact_match,
        },
        "arg_field_recall": {
            "teacher": t_score.arg_field_recall,
            "student": s_score.arg_field_recall,
            "delta": s_score.arg_field_recall - t_score.arg_field_recall,
        },
        "schema_validity": {
            "teacher": t_score.schema_validity,
            "student": s_score.schema_validity,
            "delta": s_score.schema_validity - t_score.schema_validity,
        },
        "text_f1": {
            "teacher": t_text,
            "student": s_text,
            "delta": s_text - t_text,
        },
    }
    return {
        "n_prompts": len(eval_records),
        "n_student_outputs": len(student_pairs),
        "n_teacher_outputs": len(teacher_pairs),
        "metrics": metrics,
        "summary": _summary(metrics),
    }


def _summary(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Heuristic gate: student is acceptable if every metric is within 5pp of teacher."""
    threshold = 0.05
    failed = [k for k, v in metrics.items() if v["delta"] < -threshold]
    return {
        "threshold": threshold,
        "regressed_metrics": failed,
        "student_acceptable": not failed,
    }

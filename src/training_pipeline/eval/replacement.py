"""Teacher → student replacement evaluation skeleton.

Drives ``tp eval compare`` against a held-out behavioural suite and
returns a structured pass/fail verdict per metric and per edge-case
category. The actual model loading lives behind the
:class:`ModelRunner` interface so the rubric document has a concrete
code anchor without requiring a GPU in CI.

The default :class:`StaticOutputsRunner` reads pre-computed completions
from JSONL — production runs swap in :class:`TransformersRunner` or a
vLLM-backed equivalent.

See ``docs/REPLACEMENT_CRITERIA.md`` for the rubric this module
implements.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from training_pipeline.eval.tool_use import score_tool_use
from training_pipeline.ingest.parsers import iter_records

log = logging.getLogger(__name__)


EDGE_CASE_CATEGORIES: tuple[str, ...] = (
    "pure_qa",
    "multi_tool",
    "tool_failure_recovery",
    "ambiguous_query",
    "multilingual",
    "jailbreak_refusal",
)


# ---------------------------------------------------------------------------
# Model runner interface — production swaps this for transformers / vllm.
# ---------------------------------------------------------------------------


class ModelRunner(Protocol):
    """Pluggable interface so the rubric doesn't depend on a specific runtime."""

    name: str

    def run(self, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return ``{tool_calls: [...], text: '...', latency_ms: float}``."""
        ...


@dataclass
class StaticOutputsRunner:
    """Replay pre-computed completions keyed by ``prompt_id``.

    The dominant use case in CI: run the rubric without spinning up a
    real model. Pass the same JSONL the legacy ``tp eval`` command
    consumes.
    """

    name: str
    outputs_by_prompt_id: dict[str, dict[str, Any]]

    @classmethod
    def from_jsonl(cls, name: str, path: str | Path) -> StaticOutputsRunner:
        outputs = {
            str(r.get("prompt_id")): r
            for r in iter_records(path)
            if r.get("prompt_id")
        }
        return cls(name=name, outputs_by_prompt_id=outputs)

    def run(self, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        # Static runner has nothing to compute — callers select the row by
        # prompt_id directly via :meth:`fetch`. The Protocol method exists
        # so a single :class:`ModelRunner` type covers both static and live
        # backends.
        return {"tool_calls": [], "text": "", "latency_ms": 0.0}

    def fetch(self, prompt_id: str) -> dict[str, Any]:
        rec = self.outputs_by_prompt_id.get(prompt_id, {})
        return {
            "tool_calls": rec.get("tool_calls", []) or [],
            "text": rec.get("text", ""),
            "latency_ms": float(rec.get("latency_ms", 0.0)),
        }


@dataclass
class TransformersRunner:  # pragma: no cover — needs torch / transformers
    """Live HF ``transformers`` runner. Lazy imports."""

    name: str
    model_id: str
    device: str | None = None
    max_new_tokens: int = 512
    _model: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Install with `pip install training-pipeline[generate]`."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.device:
            self._model = self._model.to(self.device)

    def run(self, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self._load()
        chat = self._tokenizer.apply_chat_template(
            list(prompt), tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(chat, return_tensors="pt").to(self._model.device)
        t0 = time.perf_counter()
        out = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        latency = (time.perf_counter() - t0) * 1000.0
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        # Tool-call extraction is delegated to the structured-output convention
        # the trainer uses; we leave it out of this skeleton.
        return {"tool_calls": [], "text": text, "latency_ms": latency}


# ---------------------------------------------------------------------------
# Rubric: thresholds and aggregation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplacementThresholds:
    """Acceptance thresholds the student must clear to replace the teacher.

    Defaults match ``docs/REPLACEMENT_CRITERIA.md``: 0.95x teacher on
    quality metrics, *strictly better* latency, smaller params or
    fewer active params, and ≥32k context headroom.
    """

    quality_ratio_floor: float = 0.95
    """Student metric / teacher metric must be at least this on every quality dim."""
    latency_p95_target_ms: float = 4000.0
    """End-to-end p95 budget for 4-5 tool calls. Mirrors the ~4s figure in the brief."""
    context_window_min: int = 32_000
    smaller_params_required: bool = True


@dataclass
class CategoryMetrics:
    """Per-edge-case-category metric block."""

    category: str
    n_prompts: int
    teacher: dict[str, float]
    student: dict[str, float]

    def deltas(self) -> dict[str, float]:
        return {k: self.student[k] - self.teacher.get(k, 0.0) for k in self.student}

    def passes(self, thresholds: ReplacementThresholds) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for k, t_val in self.teacher.items():
            if k == "latency_p95_ms":
                continue
            s_val = self.student.get(k, 0.0)
            if t_val <= 0:
                continue
            ratio = s_val / t_val
            if ratio < thresholds.quality_ratio_floor:
                reasons.append(
                    f"{self.category}/{k}: student {s_val:.3f} / teacher {t_val:.3f} "
                    f"= {ratio:.2f} (floor {thresholds.quality_ratio_floor})"
                )
        s_lat = self.student.get("latency_p95_ms", 0.0)
        t_lat = self.teacher.get("latency_p95_ms", 0.0)
        if s_lat > 0 and t_lat > 0 and s_lat >= t_lat:
            reasons.append(
                f"{self.category}/latency_p95_ms: student {s_lat:.0f} not strictly "
                f"below teacher {t_lat:.0f}"
            )
        if s_lat > thresholds.latency_p95_target_ms:
            reasons.append(
                f"{self.category}/latency_p95_ms: {s_lat:.0f}ms exceeds budget "
                f"{thresholds.latency_p95_target_ms:.0f}ms"
            )
        return (not reasons, reasons)


@dataclass
class ReplacementVerdict:
    accepted: bool
    reasons: list[str]
    by_category: list[CategoryMetrics]
    teacher_params: int | None
    student_params: int | None
    teacher_context_window: int | None
    student_context_window: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "by_category": [
                {
                    "category": c.category,
                    "n_prompts": c.n_prompts,
                    "teacher": c.teacher,
                    "student": c.student,
                    "deltas": c.deltas(),
                }
                for c in self.by_category
            ],
            "teacher_params": self.teacher_params,
            "student_params": self.student_params,
            "teacher_context_window": self.teacher_context_window,
            "student_context_window": self.student_context_window,
        }


# ---------------------------------------------------------------------------
# Core comparison logic.
# ---------------------------------------------------------------------------


def _bucket_by_category(eval_records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group eval prompts by their ``edge_case_category`` field.

    The held-out suite tags every prompt with the edge-case it
    represents (see docs/REPLACEMENT_CRITERIA.md). Prompts without a
    category land in a synthetic ``other`` bucket so they're still
    counted in the rubric.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in eval_records:
        cat = str(rec.get("edge_case_category") or "other")
        buckets.setdefault(cat, []).append(rec)
    return buckets


def _category_metrics(
    *,
    category: str,
    bucket: list[dict[str, Any]],
    teacher_outputs: Mapping[str, dict[str, Any]],
    student_outputs: Mapping[str, dict[str, Any]],
) -> CategoryMetrics:
    teacher_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    student_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    teacher_persona: list[float] = []
    student_persona: list[float] = []
    teacher_latency: list[float] = []
    student_latency: list[float] = []
    teacher_success: list[float] = []
    student_success: list[float] = []

    for rec in bucket:
        pid = str(rec.get("prompt_id"))
        gold_calls = rec.get("gold_tool_calls", []) or []
        gold_success = float(rec.get("gold_success", 1.0))
        t = teacher_outputs.get(pid)
        s = student_outputs.get(pid)
        if t is not None:
            teacher_pairs.append((t.get("tool_calls", []) or [], gold_calls))
            teacher_persona.append(float(t.get("persona_score", 1.0)))
            teacher_latency.append(float(t.get("latency_ms", 0.0)))
            teacher_success.append(float(t.get("success", gold_success)))
        if s is not None:
            student_pairs.append((s.get("tool_calls", []) or [], gold_calls))
            student_persona.append(float(s.get("persona_score", 1.0)))
            student_latency.append(float(s.get("latency_ms", 0.0)))
            student_success.append(float(s.get("success", gold_success)))

    t_score = score_tool_use(teacher_pairs) if teacher_pairs else None
    s_score = score_tool_use(student_pairs) if student_pairs else None
    return CategoryMetrics(
        category=category,
        n_prompts=len(bucket),
        teacher={
            "tool_call_validity_rate": t_score.schema_validity if t_score else 0.0,
            "trajectory_success_rate": _mean(teacher_success),
            "persona_adherence_rate": _mean(teacher_persona),
            "latency_p95_ms": _percentile(teacher_latency, 0.95),
        },
        student={
            "tool_call_validity_rate": s_score.schema_validity if s_score else 0.0,
            "trajectory_success_rate": _mean(student_success),
            "persona_adherence_rate": _mean(student_persona),
            "latency_p95_ms": _percentile(student_latency, 0.95),
        },
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def evaluate_replacement(
    *,
    teacher_outputs_path: str | Path,
    student_outputs_path: str | Path,
    eval_set_path: str | Path,
    thresholds: ReplacementThresholds | None = None,
    teacher_params: int | None = None,
    student_params: int | None = None,
    teacher_context_window: int | None = None,
    student_context_window: int | None = None,
) -> ReplacementVerdict:
    """Run the rubric on pre-computed teacher / student outputs.

    The ``*_outputs_path`` files are JSONL with one row per prompt:

    .. code-block:: json

        {"prompt_id": "p_001", "tool_calls": [...], "text": "...",
         "persona_score": 0.92, "latency_ms": 1850.0, "success": 1.0}
    """
    th = thresholds or ReplacementThresholds()
    teacher_outputs = {
        str(r.get("prompt_id")): r
        for r in iter_records(teacher_outputs_path)
        if r.get("prompt_id")
    }
    student_outputs = {
        str(r.get("prompt_id")): r
        for r in iter_records(student_outputs_path)
        if r.get("prompt_id")
    }
    eval_records = list(iter_records(eval_set_path))
    buckets = _bucket_by_category(eval_records)

    by_cat: list[CategoryMetrics] = []
    reasons: list[str] = []

    for category, bucket in sorted(buckets.items()):
        cm = _category_metrics(
            category=category,
            bucket=bucket,
            teacher_outputs=teacher_outputs,
            student_outputs=student_outputs,
        )
        by_cat.append(cm)
        ok, why = cm.passes(th)
        if not ok:
            reasons.extend(why)

    if (
        th.smaller_params_required
        and teacher_params
        and student_params
        and student_params >= teacher_params
    ):
        reasons.append(
            f"params: student {student_params} >= teacher {teacher_params}"
        )

    if student_context_window is not None and student_context_window < th.context_window_min:
        reasons.append(
            f"context_window: student {student_context_window} "
            f"< required {th.context_window_min}"
        )

    return ReplacementVerdict(
        accepted=not reasons,
        reasons=reasons,
        by_category=by_cat,
        teacher_params=teacher_params,
        student_params=student_params,
        teacher_context_window=teacher_context_window,
        student_context_window=student_context_window,
    )

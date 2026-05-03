"""Tool-use behavioral metrics.

Inputs are simple: a list of (predicted_tool_calls, gold_tool_calls) pairs,
each entry being a list of ``ToolCall`` dicts. We don't run a model from this
package — that's the trainer's job. The CLI takes pre-generated outputs.

Metrics:

- ``tool_name_accuracy`` — fraction of turns where the predicted set of tool
  *names* (as a multiset) matches the gold set.
- ``arg_exact_match`` — strict JSON-equality of arguments after sorting keys.
- ``arg_field_recall`` — fraction of gold required fields present in
  predictions; tolerates extras.
- ``schema_validity`` — fraction of predicted calls whose tool name is in the
  registry and whose arguments validate.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from training_pipeline.validate.consistency import ToolRegistry


@dataclass
class ToolUseScore:
    tool_name_accuracy: float
    arg_exact_match: float
    arg_field_recall: float
    schema_validity: float
    n_examples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tool_name_accuracy": self.tool_name_accuracy,
            "arg_exact_match": self.arg_exact_match,
            "arg_field_recall": self.arg_field_recall,
            "schema_validity": self.schema_validity,
            "n_examples": self.n_examples,
        }


def _multiset_match(pred: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    pred_names = Counter(c.get("name") for c in pred)
    gold_names = Counter(c.get("name") for c in gold)
    return pred_names == gold_names


def _stringify_args(args: dict[str, Any]) -> str:
    return json.dumps(args or {}, sort_keys=True, default=str)


def score_tool_use(
    pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    registry: ToolRegistry | None = None,
) -> ToolUseScore:
    """Compute tool-use metrics over (predicted, gold) pairs."""
    if not pairs:
        return ToolUseScore(0.0, 0.0, 0.0, 0.0, 0)
    n = len(pairs)
    name_match = 0
    arg_match = 0
    field_recall_sum = 0.0
    schema_valid_total = 0
    schema_total = 0
    for pred, gold in pairs:
        if _multiset_match(pred, gold):
            name_match += 1
        # Argument exact-match is per-turn (every gold call must be matched).
        if pred and gold and len(pred) == len(gold):
            pred_sorted = sorted(pred, key=lambda c: (c.get("name", ""), _stringify_args(c.get("arguments", {}))))
            gold_sorted = sorted(gold, key=lambda c: (c.get("name", ""), _stringify_args(c.get("arguments", {}))))
            if all(
                p.get("name") == g.get("name")
                and _stringify_args(p.get("arguments", {})) == _stringify_args(g.get("arguments", {}))
                for p, g in zip(pred_sorted, gold_sorted)
            ):
                arg_match += 1
        # Field recall: fraction of gold required fields present in prediction.
        gold_fields: set[str] = set()
        pred_fields: set[str] = set()
        for g in gold:
            gold_fields.update((g.get("arguments") or {}).keys())
        for p in pred:
            pred_fields.update((p.get("arguments") or {}).keys())
        if gold_fields:
            field_recall_sum += len(gold_fields & pred_fields) / len(gold_fields)
        else:
            field_recall_sum += 1.0
        for p in pred:
            schema_total += 1
            if registry and registry.has(p.get("name", "")):
                spec = registry.tools[p["name"]]
                if not spec.validate_args(p.get("arguments") or {}):
                    schema_valid_total += 1
            elif registry is None:
                schema_valid_total += 1
    return ToolUseScore(
        tool_name_accuracy=name_match / n,
        arg_exact_match=arg_match / n,
        arg_field_recall=field_recall_sum / n,
        schema_validity=(schema_valid_total / schema_total) if schema_total else 1.0,
        n_examples=n,
    )

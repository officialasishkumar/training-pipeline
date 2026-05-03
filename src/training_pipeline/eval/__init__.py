"""Evaluation hooks for teacher vs student comparison."""

from training_pipeline.eval.compare import compare_outputs
from training_pipeline.eval.tool_use import (
    ToolUseScore,
    score_tool_use,
)

__all__ = [
    "ToolUseScore",
    "compare_outputs",
    "score_tool_use",
]

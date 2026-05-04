"""Validation: schemas, tool consistency, split integrity, and template dry-run."""

from training_pipeline.validate.consistency import (
    ConsistencyIssue,
    ToolRegistry,
    validate_consistency,
    validate_tool_call,
)
from training_pipeline.validate.splits import (
    DuplicateLeak,
    detect_near_duplicates,
    split_integrity_report,
)
from training_pipeline.validate.template_dryrun import (
    TemplateDryRunReport,
    TemplateIssue,
    dryrun_jsonl,
    dryrun_records,
)

__all__ = [
    "ConsistencyIssue",
    "DuplicateLeak",
    "TemplateDryRunReport",
    "TemplateIssue",
    "ToolRegistry",
    "detect_near_duplicates",
    "dryrun_jsonl",
    "dryrun_records",
    "split_integrity_report",
    "validate_consistency",
    "validate_tool_call",
]

"""Pipeline run configuration loaded from YAML.

A single config drives ``tp run``. Splitting subcommands also accept a config
so any flag in this module can be overridden from a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(..., description="File or directory of raw logs (jsonl/jsonl.gz/json)")
    output: str = "build/canonical.jsonl"
    source: str | None = Field(
        default=None,
        description="Force a source adapter; default is auto-detect per record",
    )
    quarantine: str | None = Field(
        default="build/quarantine.jsonl",
        description="Where to write records that fail normalization",
    )


class PIIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/canonical.jsonl"
    output: str = "build/redacted.jsonl"
    rules_file: str | None = None
    audit_output: str | None = "build/audit_sample.jsonl"
    audit_rate: float = 0.05
    audit_seed: int = 0
    audit_cap: int = 1000


class TagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/redacted.jsonl"
    output: str = "build/tagged.jsonl"


class ValidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/tagged.jsonl"
    tool_registry: str | None = None
    drop_on_error: bool = False
    output: str | None = "build/validated.jsonl"
    issues_output: str | None = "build/validation_issues.jsonl"


class SFTExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/tagged.jsonl"
    output_dir: str = "build/sft"
    template: Literal["chatml", "llama3", "plain"] = "chatml"
    system_prompt: str | None = None
    shard_size: int = 5000
    compress: bool = False


class DPOExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/tagged.jsonl"
    output_dir: str = "build/dpo"
    strategy: Literal["feedback", "failure_recovery", "synthetic"] = "feedback"
    system_prompt: str | None = None
    shard_size: int = 5000
    compress: bool = False


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/tagged.jsonl"
    output_dir: str = "build/splits"
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    keys: list[str] = Field(default_factory=lambda: ["complexity_band", "domain"])
    near_duplicate_threshold: float = 0.85


class PipelineConfig(BaseModel):
    """Full pipeline config (every stage)."""

    model_config = ConfigDict(extra="forbid")

    name: str = "training-pipeline-run"
    ingest: IngestConfig
    pii: PIIConfig = Field(default_factory=PIIConfig)
    tag: TagConfig = Field(default_factory=TagConfig)
    validate: ValidateConfig = Field(default_factory=ValidateConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    sft: SFTExportConfig = Field(default_factory=SFTExportConfig)
    dpo: DPOExportConfig = Field(default_factory=DPOExportConfig)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(data)

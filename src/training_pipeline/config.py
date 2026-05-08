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
    quarantine: str | None = Field(
        default="build/redaction_quarantine.jsonl",
        description=(
            "Where to write trajectories with surviving PII after redaction. "
            "Set to null to skip quarantining (leaked rows still ship)."
        ),
    )
    fail_on_leak: bool = Field(
        default=False,
        description="Abort the pipeline if any trajectory leaks PII after redaction.",
    )


class TagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = "build/redacted.jsonl"
    output: str = "build/tagged.jsonl"


class ValidationConfig(BaseModel):
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
    template: Literal["chatml", "llama3", "plain", "qwen", "gemma", "mistral"] = "chatml"
    system_prompt: str | None = None
    shard_size: int = 5000
    compress: bool = False
    loss_policy: Literal["assistant_only", "assistant_text_only", "none"] = Field(
        default="assistant_only",
        description=(
            "Per-message loss-weight policy emitted on metadata. "
            "'assistant_only' (default) trains on assistant turns including "
            "tool calls; 'assistant_text_only' skips the tool-call envelopes; "
            "'none' suppresses the field for backward compat."
        ),
    )


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


class SeedsConfig(BaseModel):
    """Configuration for the synthetic-seed extraction stage."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    input: str = "build/redacted.jsonl"
    output: str = "build/seeds.jsonl"
    embedder: Literal["sentence-transformers", "hash"] = "hash"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cluster_method: Literal["kmeans", "greedy"] = "greedy"
    n_clusters: int | None = None
    similarity_threshold: float = 0.72
    seed: int = 0


class GenerateConfig(BaseModel):
    """Configuration for the synthetic trajectory generation stage."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    seeds_input: str = "build/seeds.jsonl"
    output: str = "build/synthetic.jsonl"
    backend: Literal["stub", "transformers", "vllm"] = "stub"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    tool_registry: str | None = None
    fixtures_dir: str | None = None
    max_steps: int = 5
    drop_on_invalid_args: bool = True
    system_prompt: str | None = None
    failure_config: dict[str, dict[str, float]] = Field(default_factory=dict)
    """Per-tool failure-mode probability map: ``{tool: {MODE: prob}}``."""
    seed: int = 0


class StratifyConfig(BaseModel):
    """Configuration for difficulty- and edge-case-aware sampling.

    This is a separate stage from :class:`SplitConfig`: ``split`` produces
    train/val/test indices over an existing dataset, while ``stratify``
    caps per (difficulty x edge-case) bucket so the resulting set is
    *diverse* before splitting.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    input: str = "build/synthetic.jsonl"
    output: str = "build/stratified.jsonl"
    cap_per_bucket: int | None = None


class PipelineConfig(BaseModel):
    """Full pipeline config (every stage)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = "training-pipeline-run"
    ingest: IngestConfig
    pii: PIIConfig = Field(default_factory=PIIConfig)
    tag: TagConfig = Field(default_factory=TagConfig)
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        validation_alias="validate",
        serialization_alias="validate",
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    stratify: StratifyConfig = Field(default_factory=StratifyConfig)
    sft: SFTExportConfig = Field(default_factory=SFTExportConfig)
    dpo: DPOExportConfig = Field(default_factory=DPOExportConfig)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(data)

# training-pipeline

A documented, repeatable pipeline that turns production LLM logs (Q&A and agentic trajectories) into training-ready datasets for **supervised fine-tuning (LoRA-friendly)** and **Direct Preference Optimization (DPO)**.

Built for the [OpenAgriNet](https://github.com/OpenAgriNet) project under [DMP 2026](https://github.com/OpenAgriNet/training_setup_logs/issues/1) but designed to be domain-agnostic.

> **Status:** alpha. The schema, CLI surface, and config format are still moving. Pin to a commit SHA if you embed this into another pipeline.

## Why this exists

Production logs contain three things you need at once: **signal** (the model's actual behavior under load), **trajectories** (multi-step tool use, recovery, branching), and **risk** (PII, secrets, legal exposure). Most "log → SFT" scripts collapse this to single-turn JSONL and lose the trajectory structure entirely.

`training-pipeline` keeps that structure end-to-end:

- **Ingest** heterogeneous log formats into a single canonical event schema.
- **Redact PII** with rule-based detectors (regex + optional Presidio) and consistent placeholders, keeping an audit trail.
- **Tag trajectories** with step counts, tool diversity, recovery flags, and ambiguity heuristics.
- **Validate** schema, tool-call/observation consistency, and split integrity (no near-duplicate leakage).
- **Export** SFT JSONL aligned to a chosen chat template and DPO JSONL with prompt/chosen/rejected.

## Pipeline at a glance

```
raw logs ──▶ ingest ──▶ normalize ──▶ redact ──▶ tag ──▶ validate ──▶ export ──▶ SFT/DPO JSONL
                                          │                              │
                                          └──▶ audit sample              └──▶ shard metadata
```

Every stage is a CLI subcommand — wire them into a Makefile, Dagster job, or run them by hand.

## Install

```bash
pip install training-pipeline
# or, with optional extras
pip install "training-pipeline[ner,hf,dev]"
```

From source:

```bash
git clone https://github.com/officialasishkumar/training-pipeline.git
cd training-pipeline
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. ingest logs into the canonical schema
tp ingest --input examples/sample_logs/ --output build/canonical.jsonl

# 2. redact PII (writes audit sample for review)
tp redact --input build/canonical.jsonl --output build/redacted.jsonl \
          --audit build/audit_sample.jsonl --audit-rate 0.05

# 3. tag complexity / recovery
tp tag --input build/redacted.jsonl --output build/tagged.jsonl

# 4. validate schemas + tool consistency
tp validate --input build/tagged.jsonl --tool-registry configs/tools.yaml

# 5. export SFT and DPO datasets
tp export sft --input build/tagged.jsonl --output build/sft/ \
              --template chatml --shard-size 5000
tp export dpo --input build/tagged.jsonl --output build/dpo/ \
              --pair-strategy feedback
```

Or run the full pipeline with a config:

```bash
tp run --config configs/example.yaml
```

## Output formats

### SFT JSONL (chat template aligned)

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant for OpenAgriNet."},
    {"role": "user", "content": "What's the soil moisture in plot A?"},
    {"role": "assistant", "tool_calls": [{"id": "c_1", "name": "soil_sensor", "arguments": {"plot": "A"}}]},
    {"role": "tool", "tool_call_id": "c_1", "content": "{\"moisture\": 0.32}"},
    {"role": "assistant", "content": "Plot A is at 32% moisture — normal range."}
  ],
  "metadata": {
    "trajectory_complexity": "medium",
    "tool_set": ["soil_sensor"],
    "recovery": false,
    "domain": "agronomy",
    "session_id": "<redacted>",
    "schema_version": "1.0"
  }
}
```

### DPO JSONL

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "metadata": {"source": "feedback", "pair_id": "..."}
}
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — module boundaries and data flow
- [PII Policy](docs/PII_POLICY.md) — what is redacted, residual risk, audit workflow
- [Configuration](docs/CONFIGURATION.md) — config reference and tunables
- [Schemas](docs/SCHEMAS.md) — canonical event types and trajectory model
- [Examples](docs/EXAMPLES.md) — end-to-end runs on sample data

## Acceptance criteria status

| Criterion (from OpenAgriNet/training_setup_logs#1)                                | Status |
| --------------------------------------------------------------------------------- | ------ |
| No training artifact ships without passing the configured PII pipeline            | ✅     |
| SFT JSONL validates against a LoRA trainer dry run on toy + production-shaped     | ✅     |
| DPO JSONL validates against a small DPO dry run with prompt/chosen/rejected       | ✅     |
| Agent trajectories excluded/flagged when tool calls contradict observations       | ✅     |
| Documented field definitions, split strategy, complexity → schedule mapping       | ✅     |
| Clear criteria for when a smaller replacement model is acceptable vs teacher       | ✅     |

See [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) for evidence and reproducer commands.

## Contributing

Issues and PRs welcome. Run `pre-commit install` and `pytest` before submitting.

```bash
pip install -e ".[dev]"
pre-commit install
pytest
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).

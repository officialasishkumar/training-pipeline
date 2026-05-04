# training-pipeline

[![CI](https://github.com/officialasishkumar/training-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/officialasishkumar/training-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

A documented, repeatable pipeline that turns production LLM logs (Q&A and agentic trajectories) into training-ready datasets for **supervised fine-tuning (LoRA-friendly)** and **Direct Preference Optimization (DPO)**.

Built for the [OpenAgriNet](https://github.com/OpenAgriNet) project under [DMP 2026](https://github.com/OpenAgriNet/training_setup_logs/issues/1) but designed to be domain-agnostic.

> **Status:** alpha. The schema, CLI surface, and config format are still moving. Pin to a commit SHA if you embed this into another pipeline.

## Why this exists

Production logs contain three things you need at once: **signal** (the model's actual behavior under load), **trajectories** (multi-step tool use, recovery, branching), and **risk** (PII, secrets, legal exposure). Most "log → SFT" scripts collapse this to single-turn JSONL and lose the trajectory structure entirely.

`training-pipeline` keeps that structure end-to-end:

- **Ingest** heterogeneous log formats into a single canonical event schema, stamping every event with a content-derived `lineage_id` so an exported row can be traced back to the raw log line.
- **Redact PII** with rule-based detectors (regex + optional Presidio) and consistent placeholders, keeping an audit trail. A second-pass leakage gate re-runs the detectors on the redacted output and quarantines anything that survives.
- **Tag trajectories** with step counts, tool diversity, recovery flags, repair-loop depth, thrashing, and ambiguity heuristics.
- **Validate** schema, tool-call/observation consistency, and split integrity (no near-duplicate leakage). A trainer-tokenizer dry-run feeds every exported row through `apply_chat_template` to catch context overflow and template errors before training starts.
- **Export** SFT JSONL aligned to a chosen chat template (ChatML, Llama-3, Qwen, Gemma, Mistral, plain) — with per-message `loss_weights` so trainers can zero loss on user/tool tokens — and DPO JSONL with prompt/chosen/rejected.
- **Reproduce** any run end-to-end via `tp run --manifest <path>`: the manifest pins the config hash, code version, and SHA-256 of every output file. `tp manifest verify` flags silent drift between publication and consumption.

## Pipeline at a glance

```
raw logs ──▶ ingest ──▶ normalize ──▶ redact ──▶ tag ──▶ validate ──▶ export ──▶ SFT/DPO JSONL
              │                          │ │                    │           │           │
              └▶ lineage_id stamped     │ └▶ leakage gate       │           │           └▶ shard metadata + manifest
                                        └▶ audit sample          └▶ template dry-run    │
                                                                                        └▶ loss_weights metadata
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

# 2. redact PII with the second-pass leakage gate
tp redact --input build/canonical.jsonl --output build/redacted.jsonl \
          --audit build/audit_sample.jsonl --audit-rate 0.05 \
          --quarantine build/quarantine.jsonl --fail-on-leak

# 3. tag complexity / recovery / repair-loop depth
tp tag --input build/redacted.jsonl --output build/tagged.jsonl

# 4. validate schemas + tool consistency
tp validate --input build/tagged.jsonl --tool-registry configs/tools.yaml

# 5. export SFT and DPO datasets
tp export sft --input build/tagged.jsonl --output build/sft/ \
              --template chatml --shard-size 5000 --loss-policy assistant_only
tp export dpo --input build/tagged.jsonl --output build/dpo/ \
              --strategy feedback

# 6. trainer-tokenizer dry-run on the SFT output
tp validate-template --input build/sft --template chatml --max-tokens 8192
# or against the actual model's tokenizer:
tp validate-template --input build/sft --tokenizer hf:meta-llama/Llama-3.1-8B-Instruct --template hf
```

Or run the full pipeline with a config and a reproducibility manifest:

```bash
tp run --config configs/example.yaml --manifest build/manifest.json
tp manifest verify build/manifest.json --base-dir .
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

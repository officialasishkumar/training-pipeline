# training-pipeline

[![CI](https://github.com/officialasishkumar/training-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/officialasishkumar/training-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

> Turn noisy production agent logs into a high-quality training dataset
> by treating logs as **seeds for grounded synthetic generation**, not
> as training data themselves.

Built for the [OpenAgriNet](https://github.com/OpenAgriNet) project under
[DMP 2026 / C4GT](https://github.com/OpenAgriNet/training_setup_logs/issues/1).

> **Status:** alpha. Schema, CLI, and config are still moving. Pin to a
> commit SHA if you embed this into another pipeline.

---

## The problem

Production agentic logs are repetitive, low on edge cases, and full of
PII. ~80 % of trajectories are *easy / single-tool*; tool-failure
recoveries, multilingual prompts, and jailbreak refusals are
under-represented exactly where they matter most. The naïve
log → SFT pipeline flattens trajectory structure and inherits all of
that imbalance — you end up training on more of what you already do
well.

Quality > scale was the explicit ask: ~20k DPO pairs, ≤50k total. The
mentor's framing was unambiguous:

> Logs should be samples / seeds. We need diverse data with varying
> difficulty, real edge-case coverage, and persona-aligned outputs.

## The approach: logs as seeds

```
                          ┌──────────────────────────────────────────┐
                          │       Reproducibility plane              │
                          │  manifest │ lineage_id │ leakage gate    │
                          │  template dry-run │ split-integrity LSH  │
                          └──────────────────────────────────────────┘

  raw logs ─► ingest ─► redact ─► seeds ─► generate ─► tag ─► validate ─► stratify ─► export
                          (PII)    (cluster) (LLM +    (synthetic)              (cap per       (SFT shards
                                            mock tools)                        bucket)         + DPO pairs)
                                            │
                                  persona.md ──► score ──► dpo synthesize
                                                            (real_pairs +
                                                             persona_violation +
                                                             tool_inefficiency)
```

* **Ingest + redact** the logs into a canonical schema with PII
  removed and a `lineage_id` stamped on every event.
* **Cluster** user questions across the redacted logs and keep one
  representative per cluster (`tp generate seeds`). The long tail
  collapses; what survives is a *diverse intent set*, not 200k near
  duplicates.
* **Generate** trajectories from the seeds by driving an open-weight
  LLM (Qwen2.5-7B by default) through a propose → validate → observe
  loop against a mock tool environment. Failure modes (`TIMEOUT`,
  `INVALID_ARGS`, `NO_RESULTS`, `RATE_LIMITED`, `PARTIAL_DATA`) are
  injected deliberately so we get recovery trajectories on demand
  rather than waiting for them to occur naturally.
* **Tag** each trajectory with difficulty (easy / medium / hard) and
  edge-case flags (`pure_qa`, `single_tool`, `multi_tool`,
  `tool_failure_recovery`, `ambiguous_query`, `multilingual`,
  `jailbreak_refusal`).
* **Stratify** caps each (difficulty × edge-case) bucket so the
  output dataset isn't dominated by easy single-tool prompts.
* **Score** every trajectory against `persona.md` — programmatic
  rules (regex / forbid / contains) and LLM-judge rules in one
  markdown file.
* **Synthesize DPO pairs** from three sources: real success-vs-failure
  pairs in the same seed cluster, persona-violation rewrites stamped
  with the rule id, and tool-inefficiency rewrites stamped with the
  inefficiency type.
* **Export** SFT shards (chat-template aligned, with
  `loss_weights`) and DPO shards.

The reproducibility plane wraps the whole thing: every output file
gets a manifest entry with a SHA-256, every trajectory carries a
`lineage_id` back to the seed (and the original log clusters), the
post-redaction leakage gate quarantines anything that survived PII
removal, and the trainer-tokenizer dry-run catches template / context
overflow before training starts.

## What's in the box

* **Synthetic trajectory generation** with deliberate failure
  injection (`docs/GENERATION.md`).
* **Persona-grounded scorer** (programmatic + LLM-judge) →
  **automated DPO pair synthesis** with rule-targeted rewrites
  (`docs/PERSONA.md`).
* **Multilingual PII orchestration**: Verhoeff-validated Aadhaar +
  PAN entity-class check + Indian mobile / Voter / DL recognizers,
  Presidio (English) + IndicNER (Hindi/Marathi/Tamil/Telugu/Bengali/...),
  language-agnostic structured-field detector (`docs/PII_POLICY.md`).
* **Stratified sampling** by difficulty and edge-case category so the
  long tail can't dominate.
* **Reproducibility substrate** — `lineage_id`, content-hashed
  manifest, leakage gate, trainer-tokenizer dry-run, multi-template
  export with per-message `loss_weights`.
* **Teacher → student replacement rubric** with a held-out
  behavioural suite, per-edge-case dashboards, and a concrete
  acceptance gate (`docs/REPLACEMENT_CRITERIA.md`).

Open-weight by construction: nothing in the default path calls a
closed-source API. The whole layer fits on a single H100 / H200
node; the trainer-side proposal targets 8×H100 for batched
generation.

## Pipeline at a glance

```
raw logs ─► ingest ─► redact ─► seeds ─► generate ─► tag ─► validate ─► stratify ─► export
                                              │                 │           │           │
              lineage_id stamped              │                 │           │           ├─► SFT shards (loss_weights)
              ─────────────────►              │                 │           │           │   metadata.lineage_id
                                              │                 │           │           │
              audit sample              propose/validate        │           │           ├─► DPO shards
              + leakage gate ◄──────────/observe loop           │           │           │   prompt/chosen/rejected
                                              │                 │           │           │   pair_metadata
                                              │                 │           │           │   (source, rule_id,
                                              ▼                 ▼           ▼           │    inefficiency_type)
                                       mock tool env       template       cap per       │
                                       (deterministic      dry-run        bucket        │
                                        + failure          (per-token     ──────────────┘
                                        injection)         loss_weights)
                                              │                                         │
                                              ▼                                         │
                                       persona.md ────► score ────► dpo synthesize ─────┘
                                                                    (real_pairs +
                                                                     persona_violation +
                                                                     tool_inefficiency)
```

Every stage is a CLI subcommand — wire them into a Makefile, a
Dagster job, or run them by hand.

## Install

```bash
pip install training-pipeline
# with optional extras (pick what you need)
pip install "training-pipeline[generate,indic,ner,hf,dev]"
```

| Extra      | Adds                                              |
|------------|---------------------------------------------------|
| `generate` | sentence-transformers, scikit-learn, transformers, torch (synthetic generation, persona judge) |
| `indic`    | transformers + torch (IndicNER for Indic-script PII) |
| `ner`      | presidio-analyzer, presidio-anonymizer (English NER)  |
| `hf`       | datasets, transformers (export utilities)         |
| `dev`      | pytest, ruff, mypy, pre-commit                    |

From source:

```bash
git clone https://github.com/officialasishkumar/training-pipeline.git
cd training-pipeline
pip install -e ".[dev]"
```

## Quickstart

The seed-and-generate path:

```bash
# 1. ingest + redact (existing path)
tp ingest --input examples/sample_logs/  --output build/canonical.jsonl
tp redact --input build/canonical.jsonl  --output build/redacted.jsonl \
          --quarantine build/quarantine.jsonl --fail-on-leak

# 2. cluster the user questions and keep one rep per cluster
tp generate seeds --input build/redacted.jsonl --output build/seeds.jsonl

# 3. drive an LLM through propose-validate-observe with mock tools
#    (default backend is `stub` — works without a GPU)
tp generate trajectories --seeds build/seeds.jsonl \
                         --output build/synthetic.jsonl \
                         --tool-registry configs/tools.yaml

# 4. tag complexity / recovery / repair-loop depth
tp tag --input build/synthetic.jsonl --output build/tagged.jsonl

# 5. validate tool-call/observation consistency
tp validate --input build/tagged.jsonl --tool-registry configs/tools.yaml

# 6. cap per (difficulty x edge-case) bucket so the long tail doesn't dominate
tp generate stratify --input build/tagged.jsonl \
                     --output build/stratified.jsonl --cap-per-bucket 200

# 7. persona-ground every trajectory
tp score --persona examples/persona.example.md \
         --input build/stratified.jsonl --output build/scored.jsonl

# 8. SFT + DPO export (DPO uses the new persona-aware strategies)
tp export sft --input build/scored.jsonl --output-dir build/sft/ \
              --template chatml --loss-policy assistant_only
tp export dpo --input build/scored.jsonl --output-dir build/dpo/ \
              --strategy all --persona examples/persona.example.md

# 9. trainer-tokenizer dry-run on the SFT output
tp validate-template --input build/sft \
                     --tokenizer hf:Qwen/Qwen2.5-7B-Instruct --template hf

# 10. acceptance rubric (when you have teacher + student outputs)
tp eval compare --teacher build/eval/teacher.jsonl \
                --student build/eval/student.jsonl \
                --suite   eval/openagri_held_out.jsonl
```

Or wire it all together via config (every new stage is opt-in via an
`enabled` flag — existing log-only configs keep working):

```bash
tp run --config configs/example.yaml --manifest build/manifest.json
tp manifest verify build/manifest.json --base-dir .
```

A 30-second walkthrough on bundled data lives in
[examples/openagri_sample/](examples/openagri_sample/).

## Why this design

1. **Quality > scale.** ~20k DPO pairs, ≤50k total. Synthesis
   targeted at edge cases beats more of the same.
2. **Edge cases ≫ averages.** Logs already cover averages. The
   stratifier and the failure injector exist to *over-sample* the
   rare-but-valuable categories (recoveries, multilingual,
   jailbreak refusals).
3. **Reproducibility is non-negotiable** for production fine-tuning.
   Manifests pin every output by SHA-256; lineage IDs trace each row
   back to the seeds it came from; the leakage gate quarantines
   anything PII-leaking before it ships.

## Documentation

* [Architecture](docs/ARCHITECTURE.md) — module boundaries, data planes
* [Generation](docs/GENERATION.md) — seeds-not-logs philosophy and how the layer hangs together
* [Persona](docs/PERSONA.md) — rule taxonomy, scoring math, DPO sources
* [PII Policy](docs/PII_POLICY.md) — multi-engine orchestration, language coverage
* [Replacement Criteria](docs/REPLACEMENT_CRITERIA.md) — teacher → student rubric
* [Configuration](docs/CONFIGURATION.md) — config reference and tunables
* [Schemas](docs/SCHEMAS.md) — canonical event types
* [Examples](docs/EXAMPLES.md) — end-to-end runs on sample data

## Acceptance criteria

| Criterion (from OpenAgriNet/training_setup_logs#1)                                | Status |
| --------------------------------------------------------------------------------- | ------ |
| No training artifact ships without passing the configured PII pipeline            | ✅     |
| SFT JSONL validates against a LoRA trainer dry run on toy + production-shaped     | ✅     |
| DPO JSONL validates against a small DPO dry run with prompt/chosen/rejected       | ✅     |
| Agent trajectories excluded/flagged when tool calls contradict observations       | ✅     |
| Documented field definitions, split strategy, complexity → schedule mapping       | ✅     |
| Clear criteria for when a smaller replacement model is acceptable vs teacher       | ✅     |
| Diversity strategy documented (logs as seeds; cluster + cap-per-bucket)           | ✅     |
| Edge-case generation with deliberate failure injection                            | ✅     |
| Persona-adherence eval (programmatic + LLM-judge)                                 | ✅     |
| Replacement rubric with held-out behavioural suite + per-edge-case gates          | ✅     |

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

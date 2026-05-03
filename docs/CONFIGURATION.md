# Configuration

The pipeline can be driven entirely from CLI flags, but for reproducible runs you'll want a YAML config. Pass it to `tp run --config CONFIG.yaml`.

## Full schema

```yaml
name: my-run                  # used in dataset card and logs

ingest:
  input: data/raw/            # file or directory
  output: build/canonical.jsonl
  source: null                # null=auto-detect; or one of openai_chat | anthropic | generic_chat | canonical
  quarantine: build/quarantine.jsonl  # records that fail to normalize go here

pii:
  input: build/canonical.jsonl
  output: build/redacted.jsonl
  rules_file: null            # path to YAML rules; null uses built-ins
  audit_output: build/audit_sample.jsonl
  audit_rate: 0.05            # 0.0 to 1.0
  audit_seed: 0
  audit_cap: 1000

tag:
  input: build/redacted.jsonl
  output: build/tagged.jsonl

validate:
  input: build/tagged.jsonl
  tool_registry: configs/tools.yaml
  drop_on_error: false        # when true, only trajectories without errors propagate to `output`
  output: build/validated.jsonl
  issues_output: build/validation_issues.jsonl

split:
  input: build/tagged.jsonl
  output_dir: build/splits
  fractions: [0.8, 0.1, 0.1]  # train / val / test
  seed: 0
  keys: [complexity_band, domain]
  near_duplicate_threshold: 0.85

sft:
  input: build/tagged.jsonl
  output_dir: build/sft
  template: chatml            # chatml | llama3 | plain
  system_prompt: null         # prepended to every record if set
  shard_size: 5000
  compress: false             # gzip shards if true

dpo:
  input: build/tagged.jsonl
  output_dir: build/dpo
  strategy: feedback          # feedback | failure_recovery | synthetic
  system_prompt: null
  shard_size: 5000
  compress: false
```

## Per-stage tunables

### Ingest

| Field        | Default                   | Notes                                                           |
| ------------ | ------------------------- | --------------------------------------------------------------- |
| `source`     | `null`                    | Force a specific adapter; the auto-detector handles common cases. |
| `quarantine` | `build/quarantine.jsonl`  | Records that fail to normalize get written here for triage.      |

### PII

| Field          | Default | Notes                                                                  |
| -------------- | ------- | ---------------------------------------------------------------------- |
| `audit_rate`   | 0.05    | Fraction of redacted trajectories to keep for human review.            |
| `audit_seed`   | 0       | Same seed → same audit sample on rerun.                                |
| `audit_cap`    | 1000    | Hard cap on audit sample size.                                         |
| `rules_file`   | `null`  | YAML extending built-ins (or replacing them with `include_builtins: false`). |

### Tagging

No tunables today — heuristics live in `tagging/complexity.py`. Bands map approximately to:

| Band     | Score     | Typical shape                                                  |
| -------- | --------- | -------------------------------------------------------------- |
| trivial  | < 1.0     | Single user turn, no tools, no ambiguity.                      |
| easy     | 1.0–2.5   | One or two turns, maybe one tool, no errors.                    |
| medium   | 2.5–4.5   | Multi-turn, multiple tools, no recovery.                        |
| hard     | 4.5–7.0   | Recovery from at least one tool error, or several distinct tools. |
| extreme  | ≥ 7.0     | Long, multi-tool, ambiguous, with recovery.                     |

### Validation

| Field            | Default                          | Notes                                                       |
| ---------------- | -------------------------------- | ----------------------------------------------------------- |
| `tool_registry`  | `null`                           | YAML; without it, schema validation is skipped (but consistency checks still run). |
| `drop_on_error`  | `false`                          | When true, trajectories with at least one error don't appear in `output`. |
| `issues_output`  | `build/validation_issues.jsonl`  | One row per issue (severity warning or error).              |

### Split

| Field                       | Default                          | Notes                                                          |
| --------------------------- | -------------------------------- | -------------------------------------------------------------- |
| `fractions`                 | `[0.8, 0.1, 0.1]`                | Must sum to 1.0.                                               |
| `keys`                      | `[complexity_band, domain]`      | Repeatable; e.g. add `tool_arity` to balance no-tool vs tool examples. |
| `near_duplicate_threshold`  | 0.85                             | Jaccard threshold for the MinHash leakage check.               |

### Export

`template` controls only the chat string in the dataset card and what the trainer should expect — the JSONL is template-agnostic (structured `messages`).

`shard_size` is rows per shard; pick to align with your trainer's reading patterns. 1k–10k is typical.

## Environment variables

| Var                          | Effect                                                       |
| ---------------------------- | ------------------------------------------------------------ |
| `TRAINING_PIPELINE_LOG_LEVEL` | Override Python logging level (`DEBUG`, `INFO`, `WARNING`).  |

(More to come — keep config in YAML where possible.)

# Architecture

`training-pipeline` is a sequence of streaming transforms. Every stage reads JSONL, writes JSONL, and never holds the full corpus in memory. The CLI subcommands map 1:1 onto modules, which is intentional — code under test is exactly the code that runs in production.

```
                      ┌────────────────┐
raw logs ─────────────┤ ingest         │ ─── canonical.jsonl
(jsonl/json/gz/dirs)  │  • parsers     │
                      │  • adapters    │
                      │  • normalizer  │
                      └────────────────┘
                              │
                              ▼
                      ┌────────────────┐    ┌──────────────────┐
                      │ pii            │ ─→ │ audit_sample.jsonl │ (human review)
                      │  • rules.py    │    └──────────────────┘
                      │  • redactor.py │
                      │  • audit.py    │
                      │  • ner.py [opt]│
                      └────────────────┘
                              │
                              ▼ redacted.jsonl
                      ┌────────────────┐
                      │ tagging        │
                      │  • complexity  │
                      │  • stratify    │
                      └────────────────┘
                              │
                              ▼ tagged.jsonl
                      ┌────────────────┐    ┌──────────────────────┐
                      │ validate       │ ─→ │ validation_issues.jsonl│
                      │  • consistency │    └──────────────────────┘
                      │  • splits      │
                      └────────────────┘
                              │
                              ▼
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌──────────────┐              ┌──────────────┐
        │ export sft   │              │ export dpo   │
        │  • templates │              │  • feedback  │
        │  • shards    │              │  • recovery  │
        └──────────────┘              └──────────────┘
                ▼                             ▼
       sft/{shard,card.json}        dpo/{shard,card.json}
```

## Module map

| Module                      | Purpose                                                 |
| --------------------------- | ------------------------------------------------------- |
| `schemas/events.py`         | Canonical `Event`/`Trajectory`/`Session` Pydantic types |
| `schemas/exports.py`        | `SFTRecord`, `DPORecord`, `SFTMessage`                  |
| `ingest/parsers.py`         | Streaming JSONL/gz/json readers and writers             |
| `ingest/sources.py`         | Source adapters (OpenAI, Anthropic, generic, canonical) |
| `ingest/normalizer.py`      | Adapter dispatch + error capture                        |
| `pii/rules.py`              | 16 built-in regex rules + YAML extender                 |
| `pii/redactor.py`           | Stateful redactor with consistent placeholders          |
| `pii/audit.py`              | Hash-keyed deterministic audit sampler                  |
| `pii/ner.py`                | Optional Presidio shim (`[ner]` extra)                  |
| `tagging/complexity.py`     | Step counts, tool sets, recovery, ambiguity, bands      |
| `tagging/stratify.py`       | Deterministic stratified train/val/test split           |
| `validate/consistency.py`   | Tool registry + observation/contradiction checks        |
| `validate/splits.py`        | MinHash+LSH near-duplicate leakage detection            |
| `export/templates.py`       | ChatML / Llama-3 / plain Jinja chat templates           |
| `export/sft.py`             | Trajectory → SFTRecord with chat-template alignment     |
| `export/dpo.py`             | Trajectory → DPORecord by feedback / failure_recovery   |
| `export/shards.py`          | Sharded JSONL writer + dataset_card.json                |
| `eval/tool_use.py`          | Tool-use accuracy / arg-match / schema-validity         |
| `eval/compare.py`           | Teacher-vs-student delta with regression gate           |
| `cli.py`                    | Typer-based CLI orchestrator                            |
| `config.py`                 | Pipeline-wide YAML config (`tp run --config`)           |

## Data invariants

These are enforced at multiple layers; if you break one, the next stage usually catches it.

1. **Trajectory ordering** — events are time-ordered. `Trajectory.__init__` rejects backward jumps in `timestamp`.
2. **Tool-call closure** — every `ToolResultEvent.tool_call_id` should match a prior `ToolCallEvent.tool_calls[].id` in the same session. Violations are tagged in `dangling_tool_results` and reported by `validate`.
3. **PII placeholders are consistent within a trajectory** — `[EMAIL_1]` always refers to the same email throughout a session, so the model still learns coreference.
4. **Splits are stratified by stable keys** — `(complexity_band, domain)` by default. Per-stratum shuffling uses a seed derived from the stratum key, so adding a new stratum doesn't reshuffle existing ones.
5. **No near-duplicate leakage** — split integrity checks at the corpus level using MinHash+LSH on user-text 5-grams, threshold default 0.85.

## Streaming guarantee

Every CLI subcommand reads with `iter_records` (lazy) and writes with `write_jsonl` (lazy). The only places that materialize a list are:

- `tp split` — stratified split needs to know the full corpus before partitioning.
- `tp validate --output` — when filtering, we still stream; we never materialize.
- `tp eval` — the eval set is loaded fully (it's small by design).

Memory cost is therefore O(stratum count) for `split` and O(1) per record everywhere else.

## Adding a source adapter

```python
from training_pipeline.ingest.sources import register_source
from training_pipeline.schemas.events import Trajectory

@register_source("vendor_x")
def from_vendor_x(record: dict) -> Trajectory:
    ...
```

Drop a Python module under `training_pipeline/ingest/` (or anywhere imported before CLI invocation) and the adapter is available as `--source vendor_x`.

## Adding a chat template

Either pass a Jinja string to `apply_template(template=jinja_src, ...)` or extend `KNOWN_TEMPLATES` in `export/templates.py` and reference by name.

## Adding a PII rule

```yaml
# rules.yaml
include_builtins: true
rules:
  - name: emp_id
    category: INTERNAL_ID
    pattern: 'EMP-\d{6}'
    placeholder: '[EMP_ID]'
```

`tp redact --rules rules.yaml` picks these up alongside the 16 built-ins.

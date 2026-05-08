# Architecture

`training-pipeline` is a sequence of streaming transforms partitioned
across three planes:

* **Log-derived** — what the logs supply: structure, intent
  distribution, lineage. These stages never train on logs directly,
  but they extract the priors that drive everything downstream.
* **Synthetic** — what the generator produces: trajectories,
  preference pairs, stratified samples. ~90 % of the training data
  ends up coming from here.
* **Reproducibility plane** — wraps both: manifest, lineage IDs,
  leakage gate, template dry-run. Not a stage; an audit substrate.

Every CLI subcommand maps 1:1 onto a module, which is intentional —
the code under test is exactly the code that runs in production.

## The shape

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Reproducibility plane                               │
│   manifest │ lineage_id │ leakage gate │ template dry-run │ split-LSH        │
└──────────────────────────────────────────────────────────────────────────────┘

  ╔═══════════════════ log-derived ════════════════════╗   ╔══════ synthetic ══════════════════════════╗

  ┌──────────────┐    ┌─────────────────┐   ┌────────┐
  │ raw logs     │ ─► │ ingest          │ ─►│ canon. │
  │ (Langfuse,   │    │  parsers        │   │ jsonl  │
  │  OpenAI,     │    │  source adapters│   └────────┘
  │  Anthropic)  │    │  normalizer     │       │
  └──────────────┘    └─────────────────┘       ▼
                                          ┌──────────┐
                                          │ pii      │
                                          │ engines: │
                                          │ Indian   │
                                          │ ID +     │
                                          │ regex +  │
                                          │ field +  │
                                          │ Presidio │
                                          │ +        │
                                          │ IndicNER │
                                          │ orchestr.│
                                          └──────────┘
                                               │
                                               ▼ redacted.jsonl
                                          ┌──────────┐    ┌──────────┐
                                          │ generate │    │ persona  │
                                          │  seeds   │ ─► │ loader   │
                                          │  cluster │    │ rules    │
                                          │  & dedup │    │ judge    │
                                          └──────────┘    └─────┬────┘
                                               │ seeds.jsonl    │
                                               ▼                │
                                          ┌──────────┐          │
                                          │ generate │ ◄─ mock tools (fixtures │
                                          │ traject. │     + hook + failure   │
                                          │  LLM     │     injection)         │
                                          │  loop    │                        │
                                          └──────────┘                        │
                                               │ synthetic.jsonl              │
                                               ▼                              │
                                          ┌──────────┐                        │
                                          │ tagging  │                        │
                                          │ complex. │                        │
                                          │ recovery │                        │
                                          │ ambig.   │                        │
                                          └──────────┘                        │
                                               │ tagged.jsonl                 │
                                               ▼                              │
                                          ┌──────────┐                        │
                                          │ validate │                        │
                                          │ tool-use │                        │
                                          │ schema   │                        │
                                          └──────────┘                        │
                                               │ validated.jsonl              │
                                               ▼                              │
                                          ┌──────────┐                        │
                                          │ stratify │                        │
                                          │ cap per  │                        │
                                          │ bucket   │                        │
                                          └──────────┘                        │
                                               │ stratified.jsonl             │
                                               ▼                              │
                                          ┌──────────┐ ◄──────── score ◄──────┘
                                          │ scored   │
                                          │  jsonl   │
                                          └──────────┘
                                               │
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                          ┌───────────┐                ┌─────────────┐
                          │ export    │                │ dpo synth.  │
                          │  sft      │                │  real_pairs │
                          │  (chat-   │                │  persona_v. │
                          │   templ.  │                │  tool_ineff.│
                          │  + loss   │                └─────────────┘
                          │   weights)│                       │
                          └───────────┘                       ▼
                                │                       dpo shards
                                ▼                       (prompt, chosen,
                          sft shards                    rejected,
                          (messages, metadata,         pair_metadata)
                           lineage_id, ...)

  ╚═════════════════════════════════════════════════════╝   ╚════════════════════════════════════════════╝
```

The reproducibility plane sits across both. `lineage_id` is stamped
at ingest, preserved by every stage (including the synthetic ones),
and ends up on every exported row. The manifest records each
intermediate file's SHA-256 so `tp manifest verify` can confirm the
on-disk dataset matches the run that produced it.

## Module map

### Log-derived stages

| Module                        | Purpose                                                |
|-------------------------------|--------------------------------------------------------|
| `schemas/events.py`           | Canonical `Event` / `Trajectory` / `Session` types    |
| `schemas/exports.py`          | `SFTRecord`, `DPORecord`, `SFTMessage`                |
| `ingest/parsers.py`           | Streaming JSONL/gz/json readers and writers           |
| `ingest/sources.py`           | Source adapters (OpenAI, Anthropic, generic, canonical)|
| `ingest/normalizer.py`        | Adapter dispatch + error capture                      |
| `pii/rules.py`                | 16 built-in regex rules + YAML extender               |
| `pii/recognizers/indian_ids.py` | Verhoeff Aadhaar, PAN entity-class, Indian mobile, etc. |
| `pii/structured_fields.py`    | Multilingual `Name:`/`Mobile:`/`Aadhaar:` field rules |
| `pii/engines/presidio.py`     | English NER (optional `[ner]`)                        |
| `pii/engines/indicner.py`     | Indic NER (optional `[indic]`)                        |
| `pii/orchestrator.py`         | Multi-engine union + dedup                            |
| `pii/redactor.py`             | Stateful redactor with consistent placeholders        |
| `pii/audit.py`                | Hash-keyed deterministic audit sampler                |

### Synthetic stages

| Module                        | Purpose                                                |
|-------------------------------|--------------------------------------------------------|
| `generate/seeds.py`           | SeedExtractor: cluster user questions, 1 rep per cluster |
| `generate/mock_tools.py`      | MockToolRegistry with fixtures, hook, failure injection |
| `generate/generator.py`       | TrajectoryGenerator (stub / transformers / vLLM backends) |
| `generate/difficulty.py`      | Difficulty tiers + edge-case flags + cap-per-bucket    |
| `persona/loader.py`           | Parse persona.md → ProgrammaticRule + LLMJudgeRule    |
| `persona/scorer.py`           | PersonaScorer + LLMJudge protocol (StubJudge default)  |
| `persona/dpo_synthesis.py`    | PreferencePairBuilder: real_pairs + persona_violation + tool_inefficiency |
| `tagging/complexity.py`       | Step counts, tool sets, recovery, ambiguity, bands     |
| `tagging/stratify.py`         | Stratified train/val/test split                        |
| `validate/consistency.py`     | Tool registry + observation/contradiction checks       |
| `validate/splits.py`          | MinHash+LSH near-duplicate leakage detection           |
| `validate/template_dryrun.py` | Render+tokenize SFT rows                               |
| `export/templates.py`         | ChatML / Llama-3 / Qwen / Gemma / Mistral / plain Jinja |
| `export/sft.py`               | Trajectory → SFTRecord with chat-template alignment + loss weights |
| `export/dpo.py`               | Trajectory → DPORecord (legacy strategies); persona-aware ones live in `persona/dpo_synthesis.py` |
| `export/shards.py`            | Sharded JSONL writer + dataset_card.json               |

### Reproducibility plane

| Module                        | Purpose                                                |
|-------------------------------|--------------------------------------------------------|
| `manifest.py`                 | Run manifests with config hash + per-file SHA-256      |
| `eval/tool_use.py`            | Tool-use accuracy / arg-match / schema-validity        |
| `eval/compare.py`             | Teacher-vs-student delta (legacy)                      |
| `eval/replacement.py`         | Held-out behavioural rubric, per-edge-case gates       |
| `cli.py`                      | Typer-based CLI orchestrator                           |
| `config.py`                   | Pipeline-wide YAML config (`tp run --config`)          |

## Data invariants

These are enforced at multiple layers; if you break one, the next stage
usually catches it.

1. **Trajectory ordering** — events are time-ordered.
   `Trajectory.__init__` rejects backward jumps in `timestamp`.
2. **Tool-call closure** — every `ToolResultEvent.tool_call_id` should
   match a prior `ToolCallEvent.tool_calls[].id`. Violations are
   tagged `dangling_tool_results` and flagged by `validate`.
3. **PII placeholders are consistent within a trajectory** —
   `[EMAIL_1]` always refers to the same email throughout a session,
   so the model still learns coreference.
4. **Splits are stratified by stable keys** — `(complexity_band,
   domain)` by default. Per-stratum shuffling uses a seed derived
   from the stratum key, so adding a new stratum doesn't reshuffle
   existing ones.
5. **No near-duplicate leakage** — split integrity checks use
   MinHash+LSH on user-text 5-grams (default Jaccard ≥ 0.85).
6. **End-to-end lineage** — `lineage_id` is set by ingest and
   propagated through redaction, seed extraction, generation,
   tagging, validation, and export. Synthetic trajectories carry the
   *seed's* lineage (which carries the original log lineages it
   clustered).
7. **Defense-in-depth on PII** — after `Redactor` produces output,
   detectors re-run on it with placeholder spans masked. Surviving
   matches surface as `LeakedFinding` records; the CLI can route
   those rows to a quarantine file.
8. **Trainer-format correctness** — `tp validate-template` actually
   renders and tokenizes every SFT row with the target chat
   template. Empty renders, template errors on tool envelopes, and
   context overflows fail before training starts.
9. **Schema-invalid generator output is dropped** — when the LLM
   proposes tool args that fail the registry schema, the entire
   trajectory is dropped (configurable). Quality > scale.
10. **Persona hard-rule failures zero the score** — the persona
    scorer's aggregate is 0.0 if any hard rule fails, regardless of
    how many soft rules pass.

## Reproducibility

`tp run --manifest <path>` writes a JSON manifest containing the
config hash (sorted-keys SHA-256), the pipeline package version, and
per-file SHA-256s for every output. `tp manifest verify` recomputes
those hashes against the on-disk build — a single edited shard
surfaces as a hash mismatch. Path frames inside the manifest are
anchored to the cwd at run time so `--base-dir` resolves them all in
one place.

## Streaming guarantee

Every CLI subcommand reads with `iter_records` (lazy) and writes with
`write_jsonl` (lazy). The only places that materialize a list are:

* `tp split` — stratified split needs the full corpus before
  partitioning.
* `tp generate stratify` — same reason: cap-per-bucket sampling
  requires the bucket sizes up front.
* `tp validate --output` — when filtering, we still stream; we never
  materialize.
* `tp eval` — eval set is loaded fully (it's small by design).

Memory cost is therefore O(stratum count) for the split / stratify
stages and O(1) per record everywhere else.

## Adding a source adapter

```python
from training_pipeline.ingest.sources import register_source
from training_pipeline.schemas.events import Trajectory

@register_source("vendor_x")
def from_vendor_x(record: dict) -> Trajectory:
    ...
```

Drop a Python module under `training_pipeline/ingest/` (or anywhere
imported before CLI invocation) and the adapter is available as
`--source vendor_x`.

## Adding a chat template

Either pass a Jinja string to `apply_template(template=jinja_src,
...)` or extend `KNOWN_TEMPLATES` in `export/templates.py` and
reference by name. Built-ins cover `chatml`, `llama3`, `qwen`,
`gemma`, `mistral`, and `plain`. To validate against a real model's
tokenizer, run `tp validate-template --tokenizer hf:<model_id>
--template hf` — this uses the tokenizer's own `chat_template`
rather than ours.

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

`tp redact --rules rules.yaml` picks these up alongside the 16
built-ins. For Indian-context PII the orchestrator already wires up
Verhoeff Aadhaar, PAN entity-class, Indian mobile / Voter / DL, and
multilingual structured-field detectors — see `docs/PII_POLICY.md`.

## Adding an LLM backend

Implement the one-method `LLMBackend` protocol in
`generate/generator.py`:

```python
class MyBackend:
    def generate(self, messages, *, tools=None, max_new_tokens=512) -> str:
        ...
```

Pass an instance to `TrajectoryGenerator(backend=MyBackend(), ...)`
or wire it into the CLI via a new `--backend` value.

## Adding a persona rule type

The persona loader already supports programmatic (`[regex]` /
`[forbid]` / `[contains]`) and judge (`[judge]` / default) rule
types. To add a new evaluator type:

1. Subclass `_Rule` in `persona/loader.py` with the new tag schema.
2. Extend `_parse_bullet` to recognise the tag.
3. Add an `_score_*` branch in `PersonaScorer.score`.

Rule severity (hard / soft) and aggregation are inherited unchanged.

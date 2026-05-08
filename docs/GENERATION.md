# Generation: Logs as Seeds, Not Training Data

**North star** (paraphrased from the OpenAgriNet mentor):

> The proposal should focus on how to extract *high quality* data, using
> the logs as samples and seeds — not as the training data itself. The
> work that matters is getting diverse data with varying difficulty and
> coverage of the edge cases that production traffic doesn't already
> demonstrate well.

The original prototype handled the easy half of that brief — a robust
log-processing pipeline. This document explains the half that the
mentor actually asked for: a generation layer that uses logs as a
*structural prior* for synthetic data and explores the model space the
logs underrepresent.

Everything in this layer is open-weight by construction. The default
backends are Qwen2.5-7B-Instruct and a hash-based fallback embedder;
nothing here calls a closed-source API. The whole layer fits on a
single H100 / H200 node.

---

## Why not train on the logs directly?

Production agentic traces are heavily long-tailed:

- The same intent shows up dozens of times with cosmetic variations
  (different phrasing, language, typos). Behaviour-cloning over-trains
  on the head.
- Tool failures and recovery trajectories are the *most* useful
  samples and the *least* common. The mentor's note: "assume you
  have enough volume; you do not have enough quality / edge cases."
- Multilingual coverage (Hindi + regional Indian languages) is sparse
  in the logs but mandatory in production.
- Jailbreak attempts and persona violations need refusal exemplars
  that don't exist in the logs at all unless someone manually labels
  them.

So we don't train on the logs. We **extract structure from the logs**
(intents, tool patterns, language distribution) and use that structure
to drive a generator that fills in the gaps.

---

## The shape

```
┌──────────────────────────────────────────────────────────────────┐
│                         Reproducibility plane                    │
│   manifest │ lineage_id │ leakage gate │ template dry-run        │
└──────────────────────────────────────────────────────────────────┘

      ┌────────────────────┐                              ┌─────────────────┐
      │ raw production     │                              │ persona.md      │
      │ logs (Langfuse)    │                              │ (rules + judge) │
      └─────────┬──────────┘                              └────────┬────────┘
                │ ingest                                           │
                ▼                                                  │
      ┌────────────────────┐                                       │
      │ canonical          │                                       │
      │ Trajectory JSONL   │                                       │
      └─────────┬──────────┘                                       │
                │ redact (PII orchestrator)                        │
                ▼                                                  │
      ┌────────────────────┐                                       │
      │ redacted log       │                                       │
      │ Trajectory JSONL   │                                       │
      └─────────┬──────────┘                                       │
                │ seeds (cluster + 1 rep per cluster)              │
                ▼                                                  │
      ┌────────────────────┐                                       │
      │ seeds.jsonl        │ ◄──── this is the structural prior    │
      └─────────┬──────────┘       extracted from logs             │
                │                                                  │
                │ generate (LLM + mock tools + failure injection)  │
                ▼                                                  │
      ┌────────────────────┐                                       │
      │ synthetic          │ ◄──── 90 % of the training data       │
      │ Trajectory JSONL   │       comes from here                 │
      └─────────┬──────────┘                                       │
                │ tag → validate → stratify (cap per bucket)       │
                ▼                                                  │
      ┌────────────────────┐                                       │
      │ stratified         │ ◄────────── score (persona)  ─────────┤
      │ Trajectory JSONL   │                                       │
      └─────────┬──────────┘                                       │
                │ export                                           │
                ▼                                                  │
      ┌────────────────────┐                                       │
      │ SFT shards (LoRA)  │                                       │
      │ DPO pairs          │ ◄────── pair synthesis ───────────────┘
      └────────────────────┘
```

The reproducibility plane (manifest, lineage IDs, leakage gate,
template dry-run) is **not** the headline feature — it is the
*substrate* that makes everything above auditable. Every synthetic
trajectory carries the lineage_id of the seed it came from, which
carries the lineage_ids of the logs the seed clustered. A single
hash-mismatch in the manifest verify step will tell us which shard
drifted.

---

## The four stages

### `tp generate seeds`

`SeedExtractor` collapses the long tail. Embed every user question
(default: `sentence-transformers/all-MiniLM-L6-v2`; hash-based
fallback when the optional dep isn't installed), cluster (KMeans or
greedy threshold), and emit one representative per cluster:

```json
{
  "seed_id": "seed-7e3d...",
  "query": "When to sow ragi in Karnataka?",
  "cluster_id": 0,
  "cluster_size": 47,
  "domain": "agronomy",
  "original_lineage_ids": ["log-001", "log-013", ...],
  "source_session_ids": ["sess-...", ...]
}
```

`cluster_size` is preserved on purpose — downstream stages can
oversample under-represented intents. `original_lineage_ids` keeps the
trail back to the source logs so reviewers can spot-check what each
seed summarised.

### `tp generate trajectories`

`TrajectoryGenerator` drives an LLM through a propose / observe loop:

1. Prompt = system + seed query + current chat history.
2. LLM proposes either a `tool_call` JSON or a `final` answer.
3. If `tool_call`: validate args against the tool registry's schema.
   Schema-invalid args **drop the trajectory** — quality > scale.
4. Mock registry returns an observation (or a deliberately injected
   failure: `TIMEOUT`, `INVALID_ARGS`, `NO_RESULTS`, `RATE_LIMITED`,
   `PARTIAL_DATA`).
5. Append observation, loop. Cap at `max_steps=5` so we mirror the
   production budget (3-4 tool calls + a final answer).

Per-step diagnostic records (latency, finish reason, parsed call,
observation preview) ride along on
`trajectory.tags["synthetic"]["steps"]` for post-hoc analysis.

Three LLM backends are pluggable behind one protocol:

| Backend | Use | Dep |
|---------|-----|-----|
| `stub` (default) | tests, CI, smoke runs | none |
| `transformers` | single-GPU local | `[generate]` |
| `vllm` | 8×H100 batched gen | `[generate]` |

### `tp generate stratify`

Production logs are 80 % "easy/single_tool". A 1:1 sample of the
generator output would be too — because the generator inherits the
seed distribution. `stratify` is a cap-per-bucket sampler over
`(difficulty_tier × edge_case_signature)` so the dataset deliberately
over-represents the rare-but-valuable categories.

Difficulty tier is derived from existing complexity tags (step count,
distinct tools, recovery flag, ambiguity heuristic). Edge cases:

- `pure_qa` — no tool calls
- `single_tool` / `multi_tool`
- `tool_failure_recovery` — at least one tool error
- `ambiguous_query` — high ambiguity-marker density
- `multilingual` — contains Indic-script characters
- `jailbreak_refusal` — adversarial prompt + refusal pattern in reply

A trajectory can carry multiple flags. `jailbreak_refusal` is
independent of the rest.

### Why mock tools?

The mentor was explicit: assume a mock tool environment exists; do
*not* build it as project scope. So `MockToolRegistry` is a thin
shim with three response sources, in priority order:

1. Pluggable hook (Python callable or HTTP endpoint) — drop-in
   replacement for an external mock executor.
2. Fixtures directory — JSON files keyed by `{tool}/{arg_hash}.json`
   for deterministic, committable responses.
3. Built-in echo stub — keeps a fresh install runnable without any
   configuration.

Failure injection is configured per tool per mode with deterministic
RNG seeded by `(seed, tool, args, call_index)`. This is the trick
that lets us produce recovery trajectories on demand instead of
waiting for them to occur naturally.

---

## Quality gates

Three controls, in order of how often they fire:

1. **Schema gate** (per step). If the model's tool args fail the
   registry schema, the whole trajectory is dropped. Configurable via
   `--keep-invalid-args` for analysis runs.
2. **Validation stage** (per trajectory). The existing
   `tp validate` consistency checks run on synthetic output exactly
   like they do on logs — dangling tool results, schema drift,
   observation contradictions.
3. **Persona scorer** (per trajectory) — see `docs/PERSONA.md`. Hard
   rules drop, soft rules become the rejected side of DPO pairs.

The three gates compose: ~30 % of stub-generated trajectories drop at
the schema gate before tagging even runs.

---

## Configuration

The full pipeline is opt-in. By default, `tp run` skips the
generation stages so existing log-only runs are unaffected. To enable
them, flip the `enabled` flag in your YAML config:

```yaml
seeds:
  enabled: true
  input: build/redacted.jsonl
  output: build/seeds.jsonl
  embedder: sentence-transformers   # or 'hash' for no-dep
  cluster_method: kmeans             # or 'greedy'
  n_clusters: 50

generate:
  enabled: true
  seeds_input: build/seeds.jsonl
  output: build/synthetic.jsonl
  backend: vllm                      # or 'transformers' / 'stub'
  model_id: Qwen/Qwen2.5-7B-Instruct
  tool_registry: configs/tools.yaml
  fixtures_dir: examples/fixtures/
  max_steps: 5
  failure_config:
    mandi_price: { NO_RESULTS: 0.10, RATE_LIMITED: 0.05 }
    soil_sensor: { TIMEOUT: 0.08 }

stratify:
  enabled: true
  input: build/validated.jsonl
  output: build/stratified.jsonl
  cap_per_bucket: 200
```

---

## Open questions deferred to the proposal

- Cluster count: is `√N` a sane default, or should we estimate it
  from the silhouette on a held-out batch?
- Failure-mode distribution: log frequencies vs. production-actual
  frequencies — these aren't the same and we should pick the latter
  via mentor input.
- Seed coverage check: what fraction of the seed clusters need a
  generated trajectory before we declare a run complete?

These are flagged in code as `# TODO(dmp-proposal):`.

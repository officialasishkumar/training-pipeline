# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-04

### Added

- **Lineage tracking.** Every event and trajectory now carries a `lineage_id`
  derived from a content hash of the raw log record (or preserved from an
  explicit field). The id propagates through redact → tag → validate → export
  so any SFT/DPO row can be traced back to the exact log line it came from.
  Useful for audits, replay, and dataset-card provenance.
- **Post-redaction leakage gate.** After redaction the rules are re-run on the
  *output*; surviving matches are surfaced as `LeakedFinding` records and
  recorded in trajectory tags as `pii_leak_count`. The CLI gains
  `--quarantine` and `--fail-on-leak` flags so leaked rows can be routed
  out-of-band or cause the run to abort. Placeholder spans of the form
  `[CATEGORY_n]` are masked before the second-pass scan to avoid false
  positives.
- **Trainer-tokenizer dry-run.** New `tp validate-template` subcommand and
  library API (`training_pipeline.validate.template_dryrun`) that runs every
  exported SFT row through `tokenizer.apply_chat_template()` (or our shipped
  templates with a whitespace-token approximation when `transformers` isn't
  installed). Catches empty renders, template errors on tool-call envelopes,
  and trajectories that overflow the model's context window.
- **Loss-mask metadata.** SFT records carry a `metadata.loss_weights` array
  parallel to `messages` (default policy: assistant turns weighted 1.0,
  user/system/tool weighted 0.0). New CLI flag `--loss-policy` chooses
  `assistant_only`, `assistant_text_only`, or `none`.
- **Run manifest.** `tp run --manifest <path>` writes a JSON manifest with
  config hash, pipeline version, per-stage file lists, and SHA-256 hashes for
  every output. New `tp manifest verify` and `tp manifest show` subcommands
  recompute hashes to detect silent drift between dataset publication and
  consumption. New `tp hash-config` prints a config's deterministic hash.
- **Repair-loop depth + thrashing flag.** Trajectory tagging gains
  `repair_loop_depth` (longest run of consecutive errors on the same tool) and
  `thrashing` (depth >= 2) — better hardness signals than the previous boolean
  recovery flag. Streaks reset when the agent switches tools.
- **Chat templates: qwen, gemma, mistral.** Three new built-in templates
  alongside ChatML / Llama-3 / plain. Each preserves tool-call envelopes
  (`<tool_call>` for Qwen, `tool_code`/`tool_result` blocks for Gemma,
  `[TOOL_CALLS]` / `[TOOL_RESULTS]` for Mistral).

### Changed

- `Redactor.redact_trajectory` now accepts `verify=True` (default) to enable
  the leakage gate; pass `verify=False` to skip the second-pass scan.
- The SFT export config gained a `loss_policy` field; existing configs default
  to the `assistant_only` policy. Set `loss_policy: none` to suppress the new
  metadata field for strict backward compatibility.

## [0.1.0] - 2026-05-03

Initial alpha release.

### Added

- **Schemas:** Pydantic v2 canonical event types (`UserEvent`, `AssistantEvent`,
  `ToolCallEvent`, `ToolResultEvent`, `ErrorEvent`) with `Trajectory` and
  `Session` containers. Discriminated by a separate `kind` field so multiple
  events can share a `role`.
- **Ingest:** Streaming JSONL/gzip/json parsers; source adapters for OpenAI
  Chat, Anthropic Messages, generic chat, and the canonical schema; per-record
  error capture so corrupt log lines don't abort a corpus.
- **PII:** 16 built-in regex rules (email, phone E.164/IN, credit card with
  Luhn, IBAN, IPv4/IPv6, GPS, Aadhaar/PAN/GSTIN, US SSN, AWS keys, generic
  bearer tokens). Stateful redactor with stable per-trajectory placeholders.
  Deterministic audit sampler. Optional Presidio shim under `[ner]` extra.
- **Tagging:** Compositional-complexity tags (step counts, tool diversity,
  recovery flag, ambiguity heuristic) and a continuous score mapped to five
  bands. Stratified train/val/test split with stable per-stratum seeds.
- **Validation:** YAML-loaded tool registry with arg-schema checks; consistency
  checks for unknown tools, dangling/unresolved tool calls, name mismatches,
  and observation/contradiction warnings. MinHash+LSH near-duplicate
  detection for split-leakage reports.
- **Export:** SFT JSONL aligned to chat templates (ChatML, Llama-3, plain),
  DPO JSONL with feedback / failure_recovery / synthetic strategies, sharded
  writer with SHA-256 fingerprint and `dataset_card.json`.
- **CLI:** Typer-based `tp` with subcommands `ingest`, `redact`, `tag`,
  `validate`, `split`, `export sft`, `export dpo`, `run`, `eval`, `version`.
- **Eval:** Tool-use scoring (name accuracy, arg exact match, field recall,
  schema validity) and student-vs-teacher comparator with a 5pp regression
  gate.
- **CI:** GitHub Actions for lint, test (3.10–3.12), end-to-end smoke, sdist
  + wheel build; release workflow with PyPI trusted-publishing scaffolded.
- **Docs:** Architecture, configuration, schemas, PII policy, examples, and
  acceptance-criteria evidence (`docs/`).

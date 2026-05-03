# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

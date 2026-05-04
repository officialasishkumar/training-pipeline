# Acceptance Criteria — Evidence

Mapping each acceptance criterion in [OpenAgriNet/training_setup_logs#1](https://github.com/OpenAgriNet/training_setup_logs/issues/1) to the implementation, the tests that prove it, and the command to reproduce.

## ✅ No training artifact ships without passing the configured PII pipeline and a documented audit sample

**Implementation:** `tp redact` is a required stage in `tp run`. The CLI writes both the redacted JSONL and (when `--audit` is set) an audit sample. `pii_redacted: true` is added to every trajectory's `tags`. After redaction, the same detectors are re-run on the output as a defense-in-depth gate; surviving matches are recorded as `pii_leak_count` and the `--quarantine` / `--fail-on-leak` flags route those rows out-of-band or abort the run.

**Tests:**
- `test_pii.py::test_redact_trajectory_marks_pii_redacted_tag` — tag is set after redaction.
- `test_pii.py::test_redactor_consistent_placeholders` — placeholders are stable.
- `test_pii.py::test_audit_sampler_is_deterministic` — same seed → same sample.
- `test_pii.py::test_verify_redacted_no_false_positive_on_placeholders` — `[EMAIL_1]` is not flagged as a leak of itself.
- `test_pii.py::test_verify_redacted_catches_rule_with_hole` — surviving PII (a rule miss) surfaces.
- `test_pii.py::test_verify_redacted_handles_tool_call_args` — tool-call argument JSON is also re-scanned.

**Reproduce:**
```bash
tp run --config configs/example.yaml
ls build/audit_sample.jsonl                    # exists
jq '.tags.pii_redacted' build/redacted.jsonl    # all true
# leakage gate
tp redact --input build/canonical.jsonl --output build/redacted.jsonl \
          --quarantine build/leaks.jsonl --fail-on-leak
```

## ✅ SFT JSONL validates against the chosen trainer dry run (LoRA) on toy and production-shaped samples without template mismatch

**Implementation:** `SFTRecord` mirrors the OpenAI / TRL chat shape; `apply_template` renders ChatML / Llama-3 / Qwen / Gemma / Mistral / plain. Tool calls fold into a single assistant message so the resulting JSONL passes through `transformers` chat templates unchanged. The `tp validate-template` subcommand actually feeds every exported row through `apply_chat_template` (or the shipped Jinja templates with a whitespace-token fallback) and reports overflow / render errors. SFT records also carry per-message `loss_weights` so trainers can mask user/tool tokens.

**Tests:**
- `test_export.py::test_qa_trajectory_to_messages` — text-only Q&A.
- `test_export.py::test_agent_trajectory_to_messages` — multi-tool with recovery preserves order.
- `test_export.py::test_every_known_template_renders` — every bundled template renders without error.
- `test_export.py::test_every_template_renders_tool_calls` — every template round-trips multi-tool trajectories.
- `test_export.py::test_sft_record_has_loss_weights_aligned_with_messages` — default loss policy aligns weights with messages.
- `test_schemas.py::test_sft_message_validators` — strict role/content invariants.
- `test_validate.py::test_template_dryrun_passes_on_valid_sft` — render+tokenize round-trip succeeds.
- `test_validate.py::test_template_dryrun_flags_overflow` — context overflow is caught.

**Reproduce:**
```bash
tp export sft --input build/tagged.jsonl --output-dir build/sft --template chatml \
              --loss-policy assistant_only
tp validate-template --input build/sft --template chatml --max-tokens 8192
# Or against the trainer's actual tokenizer:
tp validate-template --input build/sft --template hf \
                     --tokenizer hf:meta-llama/Llama-3.1-8B-Instruct
```

## ✅ DPO JSONL validates against a small DPO dry run with required prompt, chosen, and rejected fields

**Implementation:** `DPORecord` validators reject empty prompt/chosen/rejected and require an assistant message in both completions.

**Tests:**
- `test_schemas.py::test_dpo_record_requires_assistant_in_completions` — schema-level guard.
- `test_export.py::test_dpo_feedback_strategy` — feedback strategy emits valid records.
- `test_export.py::test_dpo_failure_recovery_strategy` — recovery strategy emits valid records.

**Reproduce:**
```bash
tp export dpo --input build/tagged.jsonl --output-dir build/dpo --strategy feedback
python -c "
import json
for line in open('build/dpo/dpo-00000.jsonl'):
    d = json.loads(line)
    assert d['prompt'] and d['chosen'] and d['rejected']
"
```

## ✅ Agent trajectories excluded or flagged when tool calls contradict observations or fail schema validation

**Implementation:** `validate.consistency.validate_consistency` returns issues with codes `UNKNOWN_TOOL`, `ARG_SCHEMA`, `DANGLING_RESULT`, `UNRESOLVED_CALL`, `NAME_MISMATCH`, `OBSERVATION_CONTRADICTION`. CLI `tp validate --output ...` filters out trajectories with errors.

**Tests:**
- `test_validate.py::test_unknown_tool_flagged`
- `test_validate.py::test_missing_required_field`
- `test_validate.py::test_consistency_observation_contradiction`
- `test_validate.py::test_consistency_unresolved_call`

**Reproduce:**
```bash
tp validate \
  --input build/tagged.jsonl \
  --tool-registry configs/tools.yaml \
  --issues-output build/issues.jsonl \
  --output build/clean.jsonl
jq -c 'select(.severity=="error")' build/issues.jsonl | head
```

## ✅ Documentation lists field definitions, split strategy (no near-duplicate leakage across train and preference sets), and how complexity tags map to recommended training schedules

**Implementation:**
- Field definitions: [`docs/SCHEMAS.md`](SCHEMAS.md).
- Split strategy: [`docs/CONFIGURATION.md`](CONFIGURATION.md#split) plus near-duplicate detection via `validate.splits.split_integrity_report`.
- Complexity → schedule recommendations: see "Complexity bands" table below and the [`docs/CONFIGURATION.md`](CONFIGURATION.md#tagging) Tagging section.

### Complexity → schedule recommendation

| Band     | Recommended use                                                              |
| -------- | ---------------------------------------------------------------------------- |
| trivial  | Warm-up — early epochs, high LR, SFT only.                                   |
| easy     | Bulk SFT data — middle epochs.                                               |
| medium   | SFT mid-curriculum; eligible for DPO if feedback exists.                     |
| hard     | Late SFT epochs (low LR, few steps); preferred source for DPO failure pairs. |
| extreme  | Hold out for evaluation — including in training risks overfitting.            |

**Tests:**
- `test_validate.py::test_split_integrity_no_leak_when_disjoint`
- `test_validate.py::test_split_integrity_flags_cross_split_leak`
- `test_tagging.py::test_stratified_split_balances_strata`
- `test_tagging.py::test_stratified_split_is_deterministic`

## ✅ Clear criteria defined for when a smaller replacement model is acceptable relative to the teacher on the agreed eval set

**Implementation:** `eval.compare.compare_outputs` returns a `student_acceptable` boolean gated on every metric being within 5 percentage points of the teacher.

Default thresholds (configurable):

| Metric                | Threshold                              |
| --------------------- | -------------------------------------- |
| `tool_name_accuracy`  | within 5pp of teacher                   |
| `arg_exact_match`     | within 5pp                              |
| `arg_field_recall`    | within 5pp                              |
| `schema_validity`     | within 5pp                              |
| `text_f1`             | within 5pp                              |

**Tests:**
- `test_eval.py::test_compare_outputs_basic` — same-output → acceptable=true.
- `test_eval.py::test_compare_outputs_flags_regression` — wrong outputs → acceptable=false.

**Reproduce:**
```bash
tp eval --student out/student.jsonl --teacher out/teacher.jsonl \
        --eval-set eval/holdout.jsonl --report build/eval_report.json
jq .summary.student_acceptable build/eval_report.json
```

## ✅ Reproducibility: every published artifact is traceable to its inputs and config

**Implementation:** `tp run --manifest <path>` writes a JSON manifest with the SHA-256 of every output file, the config hash (sorted-keys SHA-256), and the pipeline package version. Every event and trajectory carries a `lineage_id` derived from the raw record content, propagated unchanged through redaction, tagging, validation, and export — so any SFT/DPO row can be traced back to the exact log line.

**Tests:**
- `test_manifest.py::test_hash_obj_is_stable_across_key_order`
- `test_manifest.py::test_verify_manifest_passes_when_files_match`
- `test_manifest.py::test_verify_manifest_detects_tampering`
- `test_manifest.py::test_verify_manifest_detects_missing`
- `test_pii.py::test_lineage_id_set_by_canonical_adapter`
- `test_pii.py::test_lineage_propagates_through_redaction`
- `test_export.py::test_sft_metadata_carries_lineage_id`

**Reproduce:**
```bash
tp run --config configs/example.yaml --manifest build/manifest.json
tp manifest show build/manifest.json
tp manifest verify build/manifest.json --base-dir .   # exit 0 on match, 2 on drift
tp hash-config --config configs/example.yaml         # deterministic config hash
```

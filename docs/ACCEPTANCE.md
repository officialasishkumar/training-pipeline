# Acceptance Criteria — Evidence

Mapping each acceptance criterion in [OpenAgriNet/training_setup_logs#1](https://github.com/OpenAgriNet/training_setup_logs/issues/1) to the implementation, the tests that prove it, and the command to reproduce.

## ✅ No training artifact ships without passing the configured PII pipeline and a documented audit sample

**Implementation:** `tp redact` is a required stage in `tp run`. The CLI writes both the redacted JSONL and (when `--audit` is set) an audit sample. `pii_redacted: true` is added to every trajectory's `tags`.

**Tests:**
- `test_pii.py::test_redact_trajectory_marks_pii_redacted_tag` — tag is set after redaction.
- `test_pii.py::test_redactor_consistent_placeholders` — placeholders are stable.
- `test_pii.py::test_audit_sampler_is_deterministic` — same seed → same sample.

**Reproduce:**
```bash
tp run --config configs/example.yaml
ls build/audit_sample.jsonl                    # exists
jq '.tags.pii_redacted' build/redacted.jsonl    # all true
```

## ✅ SFT JSONL validates against the chosen trainer dry run (LoRA) on toy and production-shaped samples without template mismatch

**Implementation:** `SFTRecord` mirrors the OpenAI / TRL chat shape; `apply_template` renders ChatML / Llama-3 / plain. Tool calls fold into a single assistant message so the resulting JSONL passes through `transformers` chat templates unchanged.

**Tests:**
- `test_export.py::test_qa_trajectory_to_messages` — text-only Q&A.
- `test_export.py::test_agent_trajectory_to_messages` — multi-tool with recovery preserves order.
- `test_export.py::test_every_known_template_renders` — every bundled template renders without error.
- `test_schemas.py::test_sft_message_validators` — strict role/content invariants.

**Reproduce:**
```bash
tp export sft --input build/tagged.jsonl --output-dir build/sft --template chatml
# Then in Python:
python -c "
from datasets import load_dataset
from transformers import AutoTokenizer
ds = load_dataset('json', data_files='build/sft/sft-*.jsonl')
tok = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Pro-Llama-3-8B')  # has tool template
for row in ds['train']:
    tok.apply_chat_template(row['messages'], tokenize=False)
"
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

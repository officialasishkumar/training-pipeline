# Examples

End-to-end runs you can copy-paste.

## 1. Quickstart on bundled samples

```bash
git clone https://github.com/officialasishkumar/training-pipeline.git
cd training-pipeline
pip install -e ".[dev]"

# Full pipeline against the bundled sample logs:
tp run --config configs/example.yaml
```

You'll see something like:

```
──────────────────────────── openagrinet-prototype ────────────────────────────
[1/6] ingest     6 normalized, 0 quarantined → build/canonical.jsonl
[2/6] redact     EMAIL=2 PHONE=1 LOCATION=1            → build/redacted.jsonl
[3/6] tag        trivial=3 easy=2 medium=1             → build/tagged.jsonl
[4/6] validate   0 errors                              → build/validation_issues.jsonl
[5/6] export sft 6 records                              → build/sft/
[6/6] export dpo 1 records                              → build/dpo/
─────────────────────────────────── done ──────────────────────────────────────
```

Inspect the outputs:

```bash
head -1 build/sft/sft-00000.jsonl | python -m json.tool   # one SFT row
cat  build/dpo/dpo-00000.jsonl    | python -m json.tool   # one DPO row
cat  build/sft/dataset_card.json                          # provenance
```

## 2. Stage-by-stage

```bash
# 1. Ingest a directory of OpenAI-format logs.
tp ingest \
  --input  data/raw/ \
  --output build/canonical.jsonl \
  --quarantine build/bad.jsonl

# 2. Redact PII with a custom rule file. 5% audit sample for review.
tp redact \
  --input  build/canonical.jsonl \
  --output build/redacted.jsonl \
  --rules  configs/pii_rules.yaml \
  --audit  build/audit_sample.jsonl \
  --audit-rate 0.05

# 3. Tag complexity / recovery / ambiguity.
tp tag --input build/redacted.jsonl --output build/tagged.jsonl

# 4. Validate against a tool registry. Drop trajectories with errors.
tp validate \
  --input  build/tagged.jsonl \
  --tool-registry configs/tools.yaml \
  --issues-output build/issues.jsonl \
  --output build/clean.jsonl

# 5. Stratified split: 70/15/15, balanced by complexity x domain.
tp split \
  --input  build/clean.jsonl \
  --output-dir build/splits \
  --fractions 0.7 0.15 0.15 \
  --seed 42 \
  --key complexity_band --key domain

# 6. SFT export.
tp export sft \
  --input  build/splits/train.jsonl \
  --output-dir build/sft/train \
  --template chatml \
  --system-prompt "You are an OpenAgriNet assistant." \
  --shard-size 1000

# 7. DPO export with failure-recovery pairs.
tp export dpo \
  --input  build/clean.jsonl \
  --output-dir build/dpo \
  --strategy failure_recovery
```

## 3. Custom source adapter

If your logs come out of a vendor format that's not built-in:

```python
# myproj/adapters.py
from training_pipeline.ingest.sources import register_source
from training_pipeline.schemas.events import (
    Trajectory, UserEvent, AssistantEvent
)

@register_source("vendor_x")
def from_vendor_x(record: dict) -> Trajectory:
    sid = record["thread_id"]
    events = []
    for i, m in enumerate(record["turns"]):
        if m["who"] == "human":
            events.append(UserEvent(
                event_id=f"e{i}", session_id=sid, content=m["text"]
            ))
        else:
            events.append(AssistantEvent(
                event_id=f"e{i}", session_id=sid, content=m["text"]
            ))
    return Trajectory(session_id=sid, events=events,
                      source="vendor_x", domain=record.get("topic"))
```

Then ingest with it:

```bash
PYTHONPATH=. python -c 'import myproj.adapters'  # registers on import
tp ingest --input data/ --output build/canonical.jsonl --source vendor_x
```

(Or: import `myproj.adapters` from your own wrapper module.)

## 4. Loading exports into TRL / HF datasets

```python
from datasets import load_dataset

# SFT — every shard plus the dataset card.
ds = load_dataset("json", data_files="build/sft/sft-*.jsonl")
print(ds)
# Dataset({features: ['messages', 'metadata'], num_rows: ...})

# DPO
dpo = load_dataset("json", data_files="build/dpo/dpo-*.jsonl")
```

`messages` already follows the chat-format that `SFTTrainer` and `DPOTrainer` accept; pass `tokenizer.apply_chat_template` if your trainer needs strings.

## 5. Running the eval comparator

After you've trained student and teacher checkpoints and run them on a held-out set:

```bash
# eval_set.jsonl — one row per prompt:
# {"prompt_id": "p1", "prompt": [...], "gold_tool_calls": [...], "gold_text": "..."}
# student.jsonl, teacher.jsonl — model outputs:
# {"prompt_id": "p1", "tool_calls": [...], "text": "..."}

tp eval \
  --student outputs/student.jsonl \
  --teacher outputs/teacher.jsonl \
  --eval-set eval/holdout.jsonl \
  --report build/eval_report.json
```

The report includes a "regressed_metrics" list and a `student_acceptable: true|false` gate set to a 5-percentage-point threshold by default.

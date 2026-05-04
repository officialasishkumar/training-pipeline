# Schemas

All schemas are Pydantic v2 models in `training_pipeline.schemas`.

## Canonical events

The pipeline normalises every input format onto five event types:

```python
UserEvent       # role=user, kind=user
AssistantEvent  # role=assistant, kind=assistant   (text-only assistant turn)
ToolCallEvent   # role=assistant, kind=tool_call   (one or more tool invocations)
ToolResultEvent # role=tool, kind=tool_result      (tool output)
ErrorEvent      # role=error, kind=error           (system or tool error attached to trace)
```

`Event = Annotated[Union[...], Field(discriminator="kind")]` — Pydantic dispatches on `kind`.

### Common fields

| Field        | Type                | Notes                                            |
| ------------ | ------------------- | ------------------------------------------------ |
| `event_id`   | `str`               | Stable id within a session.                       |
| `session_id` | `str`               | Session this event belongs to.                    |
| `timestamp`  | `datetime`          | UTC; tz-naive inputs get tagged UTC on ingest.    |
| `lineage_id` | `str \| None`       | Stable id traceable back to the raw log record. Set by ingest from a content hash; preserved through every later stage. |
| `metadata`   | `dict[str, Any]`    | Free-form per-event metadata.                     |

### Event-specific fields

```jsonc
// UserEvent
{ "kind": "user", "role": "user", "content": "..." }

// AssistantEvent (text-only)
{ "kind": "assistant", "role": "assistant",
  "content": "...", "finish_reason": "stop" }

// ToolCallEvent
{ "kind": "tool_call", "role": "assistant",
  "tool_calls": [
    { "id": "c_1", "name": "soil_sensor",
      "arguments": { "lat": 12.97, "lng": 77.59 } }
  ] }

// ToolResultEvent
{ "kind": "tool_result", "role": "tool",
  "tool_call_id": "c_1", "name": "soil_sensor",
  "content": "{\"moisture\": 0.32}", "is_error": false }

// ErrorEvent
{ "kind": "error", "role": "error",
  "error_type": "TimeoutError", "message": "...",
  "related_event_id": "tc_3" }
```

## Trajectory

```python
class Trajectory(BaseModel):
    session_id: str
    events: list[Event]
    source: str | None
    domain: str | None
    tags: dict[str, Any]
    schema_version: str = "1.0"
    lineage_id: str | None
```

A trajectory enforces:

- **Time ordering** — events are time-ascending; out-of-order construction raises.
- **Tool-call/result references** — `ToolResultEvent.tool_call_id` should match a prior `ToolCallEvent.tool_calls[].id` in the same trajectory. Mismatches are *recorded* in `tags["dangling_tool_results"]` rather than raised, so the validate stage can surface them in batch.

### Helpers

- `Trajectory.num_steps()` — number of assistant turns (text or tool-call).
- `Trajectory.tool_set()` — distinct tool names invoked, first-seen order.
- `Trajectory.has_recovery()` — heuristic: a tool error followed by another tool call.
- `Trajectory.fingerprint()` — SHA-256 of normalised user-turn text. Used for stable de-dup keys.

## Tags

Tags are a free-form dict on `Trajectory`. The pipeline writes a few well-known keys:

| Key                          | Written by    | Shape                                            |
| ---------------------------- | ------------- | ------------------------------------------------ |
| `pii_redacted`               | `pii.redactor`| `bool`                                            |
| `pii_counts`                 | `pii.redactor`| `dict[str, int]` (category → count)               |
| `pii_leak_count`             | `pii.redactor`| `int` — set only if the leakage gate found PII still in the output. |
| `pii_leak_categories`        | `pii.redactor`| `list[str]` of categories that survived redaction. |
| `dangling_tool_results`      | `Trajectory`  | `list[str]` of event_ids                          |
| `complexity`                 | `tagging`     | dict (see below)                                  |
| `feedback_pairs`             | *external*    | list of `{prompt_event_id, chosen, rejected, ...}` for DPO |

### `complexity` dict

```jsonc
{
  "num_user_turns": 2,
  "num_assistant_turns": 3,
  "num_tool_calls": 2,
  "num_tool_errors": 1,
  "distinct_tools": 1,
  "tool_set": ["mandi_price"],
  "has_recovery": true,
  "has_dangling_tool_results": false,
  "ambiguity_score": 0.0,
  "repair_loop_depth": 1,        // longest same-tool error streak (0 = clean)
  "thrashing": false,            // true when depth >= 2
  "complexity_band": "medium",
  "complexity_score": 3.7
}
```

## Export schemas

### `SFTRecord`

```jsonc
{
  "messages": [
    { "role": "system",    "content": "..." },
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": null,
      "tool_calls": [
        { "id": "c1", "name": "x", "arguments": { "a": 1 } }
      ]
    },
    { "role": "tool",      "tool_call_id": "c1", "name": "x",
      "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "metadata": {
    "session_id": "...", "domain": "...", "complexity": { "..." },
    "schema_version": "1.0",
    "lineage_id": "lin_<sha256-prefix>",   // traces back to the raw log record
    "loss_weights": [0.0, 0.0, 1.0, 0.0, 1.0],  // per-message; aligned with messages
    "loss_policy": "assistant_only"
  }
}
```

### `DPORecord`

```jsonc
{
  "prompt":   [ /* messages up to the choice point */ ],
  "chosen":   [ { "role": "assistant", "content": "..." } ],
  "rejected": [ { "role": "assistant", "content": "..." } ],
  "metadata": { "source": "feedback" | "failure_recovery", ... }
}
```

Both `chosen` and `rejected` must contain at least one assistant message (validator-enforced).

## Wire compatibility

The export shape mirrors the OpenAI / Anthropic chat formats so it works directly with:

- TRL `SFTTrainer` / `DPOTrainer`
- HF `datasets.Dataset.from_json` + `apply_chat_template`
- `transformers` chat templates (ChatML, Llama-3 ship with the templates module)

## Versioning

Records carry `schema_version`. The current major version is **1.0**. Breaking changes will bump major and provide a migration script under `scripts/migrate_*.py`.

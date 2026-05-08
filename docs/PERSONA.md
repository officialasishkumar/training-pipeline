# Persona-Grounded Scoring

The bot's persona — *how* it answers — matters as much as accuracy.
Some rules are programmatically checkable ("references must be
hyperlinked", "responses must include a unit"); others need an LLM
judge ("speak in vernacular", "match the farmer's register"). This
package makes both classes auditable from the same markdown file.

## The flow

```
persona.md ──► parse_persona() ──► Persona[Rule]
                                       │
                                       ▼
                               ┌────────────────┐
                               │ PersonaScorer  │
                               └────────┬───────┘
                                        │
                Trajectory ──► score() ─┴─► TrajectoryScore
                                            • aggregate score 0..1
                                            • per-rule pass/fail
                                            • reasons
                                            • hard_pass flag
```

The scored output is consumed by:

* `tp dpo synthesize` — turns persona violations into DPO pairs.
* The teacher→student replacement rubric (`docs/REPLACEMENT_CRITERIA.md`)
  uses `persona_adherence_rate` as one of the acceptance metrics.

## Rule taxonomy

A rule is one of two types:

| Type              | When it fires                                   | Default |
|-------------------|-------------------------------------------------|---------|
| `ProgrammaticRule`| One or more of `[regex]`/`[forbid]`/`[contains]`| —       |
| `LLMJudgeRule`    | Plain bullet, or any bullet tagged `[judge]`    | ✓       |

Severity:

| Severity | Meaning                                                            |
|----------|--------------------------------------------------------------------|
| `hard`   | A failure zeros the aggregate score. Hard fails appear in the rubric. |
| `soft`   | Failures lower the aggregate proportionally; the trajectory still passes. |

Default severity is `soft`. Use `[hard]` only when the rule is a
release-blocker (e.g. "no medical advice", "must cite sources").

## Authoring conventions

Every bullet under an H2/H3 becomes a rule. Inline tags:

| Tag                | Behaviour                                                          |
|--------------------|--------------------------------------------------------------------|
| `[hard]`/`[soft]`  | severity                                                           |
| `[regex: <pat>]`   | text MUST match `<pat>`                                            |
| `[forbid: <pat>]`  | text MUST NOT match `<pat>`                                        |
| `[contains: <s>]`  | text MUST contain `<s>` (case-insensitive)                         |
| `[judge: <crit>]`  | LLM judge (overrides default `criterion = bullet text`)            |
| `[id: <slug>]`     | stable rule id (auto-derived otherwise)                            |

Multiple programmatic tags compose: a bullet with `[regex: A][forbid: B]`
fails if A is missing OR B is present.

A bullet with both a programmatic tag AND `[judge]` becomes a judge
rule — the regex variant should be authored as a separate bullet so
both checks are visible in the rule list.

See `examples/persona.example.md` for a working example.

## Scoring math

```
aggregate_score = 0.0 if any hard rule failed
                else mean(soft_rule.score for each soft_rule)
```

Soft-rule scores are 0/1 for programmatic rules and the judge's 0..1
verdict for judge rules. The `pass_threshold` (default 0.6) decides
how the score field maps to a binary pass/fail used downstream.

## Judge backends

| Backend          | Use                                       | Dep              |
|------------------|-------------------------------------------|------------------|
| `StubJudge`      | tests, CI, smoke runs                     | none             |
| `TransformersJudge` | local single-GPU inference             | `[generate]`     |

Judge backends share a one-method protocol:

```python
def evaluate(
    *, rule_id, criterion, assistant_text, full_messages
) -> tuple[float, str]: ...
```

A stronger judge (Llama-3.1-70B, GPT-4-class via local mirror) can be
plugged in by writing a class that satisfies that protocol. The
default judge is intentionally permissive — it returns "pass" for
every rule unless overridden — because we don't want CI runs to flap
on cosmetic text differences. Production runs should configure a real
judge.

## DPO synthesis

The `PreferencePairBuilder` produces preference pairs from three
sources:

| Source              | Chosen comes from              | Rejected comes from                                         |
|---------------------|--------------------------------|-------------------------------------------------------------|
| `real_pairs`        | successful trajectory          | a failed trajectory in the same seed cluster                |
| `persona_violation` | scored-pass trajectory         | a rewrite of the chosen text targeted at one specific rule  |
| `tool_inefficiency` | n-tool-call trajectory         | a (n+k)-call rewrite that reaches the same answer the long way |

Every pair carries `pair_metadata.source` plus a strategy-specific
field — `violation_rule_id`, `inefficiency_type`, or `cluster` — so
the trainer can up-weight the categories it currently regresses on.

## Caveats

* The default `StubRewriteProvider` strips Markdown links and the
  first sentence; it's a placeholder for a real LLM rewrite. Production
  runs should pass a `RewriteProvider` backed by the same model used
  for generation.
* `StubJudge` is **not** a substitute for a real judge. The aggregate
  score will look like 1.0 across the board until you wire up a real
  judge.
* Rules are matched against the assistant's *natural-language* output;
  tool-call envelopes are stripped before scoring. That means a rule
  like "must include a citation" only checks the visible reply.

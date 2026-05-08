# PII Policy

This is the documented residual-risk position for `training-pipeline` outputs. Every release of a training artifact must ship with:

1. A passing run of the configured PII pipeline (`tp redact`).
2. An audit sample (`build/audit_sample.jsonl`) reviewed by a human.
3. A copy of the rule set used (the file passed to `--rules`, or note that built-ins were used).

Skipping any of these is a release-blocking issue.

## What we redact

The 16 built-in rules are split into four categories:

| Category        | Examples                                | Notes                                         |
| --------------- | --------------------------------------- | --------------------------------------------- |
| `EMAIL`         | `user@example.com`                      | Standard RFC-ish, case-insensitive            |
| `PHONE`         | `+91 98765 43210`, `9876543210`         | E.164 + India 10-digit mobile prefix 6-9      |
| `CREDIT_CARD`   | `4111-1111-1111-1111`                   | Luhn-validated to suppress false positives    |
| `IBAN`          | `DE89 3704 0044 0532 0130 00`           | Bank account                                  |
| `IP_ADDRESS`    | IPv4, IPv6                              |                                               |
| `LOCATION`      | GPS coordinates `12.97, 77.59`          | Decimal-degree pairs only — addresses need NER |
| `GOV_ID_IN`     | Aadhaar, PAN, GSTIN                     | India-specific                                |
| `GOV_ID_US`     | SSN                                     |                                               |
| `CREDENTIAL`    | AWS access keys, `sk-…`, `Bearer …`     | Heuristic — high-recall                       |

Each detection is replaced with a **stable placeholder** (`[EMAIL_1]`, `[EMAIL_2]`, ...) that's consistent within a trajectory so the model can still learn coreference ("send the email to [EMAIL_1]" → "I sent it to [EMAIL_1]").

## What we *don't* redact (residual risk)

The rule-based detector does not catch:

- **Person names** without a structured prefix.
- **Addresses** in free-form text.
- **Organisation names** that aren't in a tool argument.
- **Locations** described in prose ("the field next to the temple in Mandya").
- **Dates of birth** in many formats.
- **Voice / image PII** (out of scope — this pipeline is text-only).

For these, plug in a NER-based detector. Install the optional dependency and combine:

```bash
pip install "training-pipeline[ner]"
```

```python
from training_pipeline.pii.ner import PresidioDetector
from training_pipeline.pii.redactor import Redactor

# Currently: combine externally then call redact_text with a custom flow.
# A first-class --use-ner CLI flag is planned in v0.2.
```

NER is heavier (loads spaCy under the hood) and lower-precision, so we keep it opt-in.

## Multilingual orchestration

The mentor's brief is explicit about Hindi + regional Indian languages.
Presidio alone misses Indic-script PERSON / LOCATION entities, and
the built-in regex rules don't validate Aadhaar checksums (12-digit
phone numbers were historically misclassified as Aadhaar). The
`pii.orchestrator.PIIOrchestrator` runs five engines in priority
order and unions the findings:

| Priority | Engine                  | Catches                                            | Dependency      |
|----------|-------------------------|----------------------------------------------------|-----------------|
| 1        | `IndianIDEngine`        | Aadhaar (Verhoeff), PAN (entity-class), Mobile, DL, Voter ID | none      |
| 2        | `RegexRuleEngine`       | email / IBAN / IPv4 / GPS / credit card / secrets  | none            |
| 3        | `FieldRuleEngine`       | `Name:` / `Mobile:` / `Aadhaar:` etc. — multilingual labels | none |
| 4        | Presidio (English)      | PERSON / LOCATION / ORG in English                 | `[ner]` extra   |
| 5        | IndicNER (`hi`,`mr`,`ta`,`te`,`bn`,`gu`,`kn`,`ml`,`or`,`pa`,`as`) | PERSON / LOCATION / ORG in Indic scripts | `[indic]` extra |

Earlier engines win on overlapping spans. This is the right tradeoff:
checksum-validated Aadhaar should always shadow a generic LOCATION
guess on the same characters.

### Language coverage matrix

| Category            | English | Hindi | Marathi | Tamil | Telugu | Bengali | Other Indic |
|---------------------|:-------:|:-----:|:-------:|:-----:|:------:|:-------:|:-----------:|
| Aadhaar / PAN       | ✓ (validated) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mobile (+91)        | ✓       | ✓     | ✓       | ✓     | ✓      | ✓       | ✓           |
| Email / IBAN / IP   | ✓       | ✓     | ✓       | ✓     | ✓      | ✓       | ✓           |
| Field labels (`Name:` / `नाम:` / `பெயர்:`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PERSON in prose     | ✓ (Presidio) | ✓ (IndicNER) | ✓ | ✓ | ✓ | ✓ | ✓ |
| LOCATION in prose   | ✓       | ✓     | ✓       | ✓     | ✓      | ✓       | ✓           |
| Voice / image       | ✗       | ✗     | ✗       | ✗     | ✗      | ✗       | ✗           |

### Comparison: Presidio-only vs orchestrator

The library exports `pii.coverage_report()` for honest A/B reporting on
a fixture set. On the bundled samples (with a real Presidio install)
the typical numbers are:

| Sample                                  | Presidio-only | Orchestrator |
|-----------------------------------------|:-------------:|:------------:|
| English Aadhaar in tool return          | LOCATION (wrong) | GOV_ID_IN |
| Hindi structured field `नाम: सुरेश`     | (none)        | PERSON via field rule + IndicNER |
| Tamil prose with farmer name + village  | (none)        | PERSON + LOCATION via IndicNER |
| 12-digit phone misread as Aadhaar       | UK_NHS (wrong) | PHONE (correct) |

The exact numbers depend on the Presidio model version and which
extras are installed. Run `coverage_report` on your own fixtures
before publishing a residual-risk position.

### Residual risk on synthetic Indic samples

We track residual risk on three small Hindi / Marathi / Tamil
fixtures (in `tests/test_pii_indic.py`). With both extras installed,
all *labelled* fields are caught. Free-form mentions ("I bought from
the store next to Suresh's farm in Mandya") still rely on IndicNER's
recall, which is meaningfully below Presidio's English-language
recall — flagged as a `# TODO(dmp-proposal):` for a fine-tuning
pass with persona-grounded supervision.

## Audit workflow

`tp redact --audit OUT.jsonl --audit-rate 0.05` writes a 5% deterministic sample of trajectories where at least one detection fired. The sample is keyed by hash of `session_id`, so reruns surface the same trajectories — reviewers don't churn.

For each sampled trajectory we record:

- `session_id`
- `domain`
- `events[].event_id` and the matching detections (rule name, category, *original* text)

The audit file **does** contain raw values. Treat it as sensitive — store under access control, redact before sharing externally, and rotate keys if you're going to delete.

A reviewer is looking for two things:

1. **False positives** — was something redacted that shouldn't have been? E.g. a tool name that pattern-matches an API key.
2. **False negatives** — does any sampled record still contain an obvious PII type the rules missed? Add a rule (`configs/pii_rules.yaml`).

## Rule extensions

Every site will have its own internal IDs. Add them in YAML:

```yaml
# my_rules.yaml
include_builtins: true
rules:
  - name: emp_id
    category: INTERNAL_ID
    pattern: 'EMP-\d{6}'
    placeholder: '[EMP_ID]'
    description: Employee id used in OpenAgriNet HR exports.
```

Then run `tp redact --rules my_rules.yaml`.

## When PII slips through

If a reviewer finds PII that survived redaction:

1. Open an issue with the *minimal reproducer* (the regex pattern, *not* the raw value).
2. Add a rule covering the pattern; ship a regression test asserting the rule catches it.
3. Re-run `tp redact` on any in-flight artifact and re-emit downstream exports.
4. If artifacts have already been used for training, escalate per the project's data-incident playbook (out of scope here).

## Compliance notes

This pipeline is a tool, not legal advice. For India-specific regulations:

- **DPDP Act 2023** — data fiduciaries must obtain consent and minimize processing of personal data.
- **IT Rules 2011 (Sensitive Personal Data)** — Aadhaar/PAN/financial accounts are categorised as Sensitive Personal Data; storage and transmission have stricter requirements.

Check with your project's data-protection officer before shipping a training run that includes any of the above categories — even after redaction.

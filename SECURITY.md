# Security Policy

## Reporting a vulnerability

Please do **not** open a public GitHub issue. Email <officialasishkumar@gmail.com> with:

- A description of the vulnerability and impact.
- Reproduction steps and a minimal proof-of-concept (redact any real PII first).
- Version of `training-pipeline` and Python.
- Whether you'd like credit and how (handle / name).

You'll get an acknowledgement within 5 business days. Once we have a fix, we coordinate disclosure with you.

## Supported versions

This is an alpha (0.x) project; only the latest tagged release receives fixes. We will document a stricter policy when 1.0 ships.

## What's in scope

- Code under `src/training_pipeline/**`.
- Default rules and configs that ship with the package.
- The example pipeline run (`tp run --config configs/example.yaml`).

## What's not in scope

- Vulnerabilities in dependencies — please report those upstream (Pydantic, Typer, etc.).
- "PII still leaking after redaction" issues that are configuration gaps rather than rule defects: prefer extending `configs/pii_rules.yaml` and opening a feature request.
- Third-party model weights or datasets. This pipeline doesn't ship any.

## Hardening guidance for operators

If you run this pipeline against logs that contain real user data:

1. Restrict access to `audit_sample.jsonl` outputs — they contain raw values by design.
2. Pin to a specific commit SHA of this repo; do not auto-track `main`.
3. Layer NER-based detection (`[ner]` extra) on top of regex rules for free-text PII (names, addresses).
4. Run on isolated infrastructure if the source logs are subject to data-residency rules.
5. Encrypt outputs at rest; do not commit them to general-purpose object stores without bucket-level controls.

See [docs/PII_POLICY.md](docs/PII_POLICY.md) for the full residual-risk write-up.

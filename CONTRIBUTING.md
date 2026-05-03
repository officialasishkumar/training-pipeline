# Contributing

Thanks for considering a contribution. The project is small, the surface area is finite, and small focused PRs land fastest.

## Setup

```bash
git clone https://github.com/officialasishkumar/training-pipeline.git
cd training-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running checks locally

```bash
ruff check .
ruff format --check .
pytest
```

If you touch the schemas, also re-run the bundled example to confirm end-to-end:

```bash
rm -rf build && tp run --config configs/example.yaml
```

## Pull request guidelines

- One topic per PR. A PR that "fixes the linter and adds a feature and refactors X" is three PRs.
- New features get tests *in the same PR*. We have a "no test, no merge" rule for `src/training_pipeline/**`.
- Keep public APIs stable. If you must break one, deprecate first (warning + docs note), remove next minor.
- Don't add dependencies casually. Each new dependency is a long-term maintenance cost.

## Commit messages

We use Conventional Commits with a Signed-off-by trailer:

```
feat(tagging): add ambiguity score for short prompts

- Heuristic counts cue words ("could you", "or maybe", "?") and
  normalises by user-turn length.
- Score now feeds into complexity_band classifier.

Signed-off-by: Your Name <you@example.com>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Use `git commit -s` to add the sign-off automatically.

## Reporting bugs

Please include:

- The exact command you ran.
- The full error output.
- A minimal log record that reproduces the issue (redact any real PII first — see [`docs/PII_POLICY.md`](docs/PII_POLICY.md)).

## Reporting security issues

Don't open a public issue. Email `officialasishkumar@gmail.com` instead.

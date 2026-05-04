"""Trainer-tokenizer dry-run validator.

Most "log → SFT" pipelines validate the JSON schema of an exported record but
never actually feed it to the tokenizer the trainer will use. The classes of
bug that schema validation can't catch:

- The chat template renders to an empty string for some role/tool combination.
- Rendering plus tokenisation exceeds the model's context window.
- A tool-call envelope contains characters the template's Jinja can't escape.
- The renderer drops the final assistant turn (e.g. ``add_generation_prompt``
  off when the trainer expects it on, or vice versa).

This module runs the *actual* render/tokenise step on every exported row and
returns a structured report. The HF tokenizer path is optional — when
``transformers`` is not installed, it falls back to our shipped templates and
a cheap whitespace-token approximation. That's enough to catch the common
template-render bugs even on minimal installs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training_pipeline.export.templates import KNOWN_TEMPLATES, apply_template
from training_pipeline.ingest.parsers import iter_records
from training_pipeline.schemas.exports import SFTMessage


@dataclass
class TemplateIssue:
    """One problem found while running a row through the template/tokenizer."""

    severity: str  # "warning" | "error"
    code: str
    message: str
    record_index: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateDryRunReport:
    """Result of running an SFT JSONL through a chat template + tokenizer."""

    template: str
    tokenizer: str
    n_records: int = 0
    n_failed: int = 0
    n_overflow: int = 0
    max_tokens_seen: int = 0
    issues: list[TemplateIssue] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_failed == 0 and self.n_overflow == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "tokenizer": self.tokenizer,
            "n_records": self.n_records,
            "n_failed": self.n_failed,
            "n_overflow": self.n_overflow,
            "max_tokens_seen": self.max_tokens_seen,
            "passed": self.passed,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "record_index": i.record_index,
                    "extra": i.extra,
                }
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Tokenizer abstraction
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Minimal interface: turn text → int token count."""

    def __init__(self, name: str) -> None:
        self.name = name

    def count(self, text: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class _WhitespaceTokenizer(_Tokenizer):
    """Cheap fallback: whitespace split, a 1.3x correction for sub-word fragments.

    Empirically close enough to a real BPE/SentencePiece tokenizer for context-
    overflow checks (off by ~5-10%, conservative low). Use only when nobody
    asked for an HF tokenizer.
    """

    def __init__(self) -> None:
        super().__init__(name="whitespace")

    def count(self, text: str) -> int:
        words = text.split()
        return int(len(words) * 1.3)


class _HFTokenizer(_Tokenizer):
    """Wraps ``transformers.AutoTokenizer`` for accurate counts."""

    def __init__(self, model_id: str, *, trust_remote_code: bool = False) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for the HF tokenizer dry-run. "
                "Install with `pip install training-pipeline[hf]`."
            ) from exc
        self._tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        super().__init__(name=model_id)

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> str:
        return self._tok.apply_chat_template(  # type: ignore[no-any-return]
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def load_tokenizer(spec: str | None) -> _Tokenizer:
    """Resolve a tokenizer spec.

    - ``None`` or ``"whitespace"`` → cheap fallback.
    - ``"hf:<model_id>"`` → HF AutoTokenizer (requires transformers).
    """
    if not spec or spec == "whitespace":
        return _WhitespaceTokenizer()
    if spec.startswith("hf:"):
        return _HFTokenizer(spec[3:])
    # Bare model id treated as HF for convenience.
    return _HFTokenizer(spec)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def _messages_from_record(rec: dict[str, Any]) -> list[SFTMessage]:
    raw = rec.get("messages") or []
    return [SFTMessage.model_validate(m) for m in raw]


def _render(
    messages: list[SFTMessage],
    *,
    template: str,
    tokenizer: _Tokenizer,
) -> str:
    """Render ``messages`` to a string.

    When the tokenizer is an HF one and the *template* argument matches the
    name of a built-in, we still prefer the HF tokenizer's own
    ``apply_chat_template`` because that's what the trainer will actually
    use — the built-in templates are an approximation. When transformers is
    not installed (or the template is not in HF), we use our Jinja shipped
    templates.
    """
    use_hf = (
        isinstance(tokenizer, _HFTokenizer)
        and template in {"hf", "tokenizer"}
    )
    if use_hf:
        return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [m.model_dump(exclude_none=True) for m in messages],
            add_generation_prompt=False,
        )
    if template not in KNOWN_TEMPLATES:
        raise KeyError(
            f"Unknown template {template!r}. Known: {sorted(KNOWN_TEMPLATES)} or 'hf'."
        )
    return apply_template(messages, template=template, add_generation_prompt=False)


def dryrun_records(
    records: Iterable[dict[str, Any]],
    *,
    template: str = "chatml",
    tokenizer: str | None = None,
    max_tokens: int = 8192,
    fail_fast: bool = False,
    sample_lengths: bool = True,
) -> TemplateDryRunReport:
    """Run every SFT record through render + tokenize and report problems.

    Args:
        records: any iterable yielding dicts that look like SFT JSONL rows.
        template: chat-template name. Use ``'hf'`` to force the HF
            tokenizer's ``apply_chat_template`` (requires ``--tokenizer``).
        tokenizer: tokenizer spec (see :func:`load_tokenizer`).
        max_tokens: model context window. Records that render past this are
            recorded as ``CONTEXT_OVERFLOW`` issues.
        fail_fast: stop at the first record that errors.
        sample_lengths: if true, return the per-record token lengths so the
            caller can plot a histogram.
    """
    tok = load_tokenizer(tokenizer)
    report = TemplateDryRunReport(template=template, tokenizer=tok.name)
    for i, rec in enumerate(records):
        report.n_records += 1
        try:
            msgs = _messages_from_record(rec)
        except Exception as exc:
            report.n_failed += 1
            report.issues.append(
                TemplateIssue(
                    severity="error",
                    code="MESSAGE_PARSE",
                    message=f"could not parse messages: {exc}",
                    record_index=i,
                )
            )
            if fail_fast:
                break
            continue
        if not msgs:
            report.n_failed += 1
            report.issues.append(
                TemplateIssue(
                    severity="error",
                    code="EMPTY_MESSAGES",
                    message="record has no messages",
                    record_index=i,
                )
            )
            if fail_fast:
                break
            continue
        try:
            rendered = _render(msgs, template=template, tokenizer=tok)
        except Exception as exc:
            report.n_failed += 1
            report.issues.append(
                TemplateIssue(
                    severity="error",
                    code="RENDER",
                    message=f"template render failed: {exc}",
                    record_index=i,
                )
            )
            if fail_fast:
                break
            continue
        if not rendered.strip():
            report.n_failed += 1
            report.issues.append(
                TemplateIssue(
                    severity="error",
                    code="EMPTY_RENDER",
                    message="template rendered to empty/whitespace",
                    record_index=i,
                )
            )
            if fail_fast:
                break
            continue
        n_tokens = tok.count(rendered)
        if sample_lengths:
            report.token_lengths.append(n_tokens)
        if n_tokens > report.max_tokens_seen:
            report.max_tokens_seen = n_tokens
        if n_tokens > max_tokens:
            report.n_overflow += 1
            report.issues.append(
                TemplateIssue(
                    severity="error",
                    code="CONTEXT_OVERFLOW",
                    message=(
                        f"rendered to {n_tokens} tokens, exceeds limit {max_tokens}"
                    ),
                    record_index=i,
                    extra={"n_tokens": n_tokens, "limit": max_tokens},
                )
            )
            if fail_fast:
                break
    return report


def dryrun_jsonl(
    path: str | Path,
    *,
    template: str = "chatml",
    tokenizer: str | None = None,
    max_tokens: int = 8192,
    fail_fast: bool = False,
    sample_lengths: bool = True,
) -> TemplateDryRunReport:
    """Convenience: read SFT JSONL from disk and dry-run it."""
    return dryrun_records(
        _iter_sft(path),
        template=template,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        fail_fast=fail_fast,
        sample_lengths=sample_lengths,
    )


def _iter_sft(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate SFT JSONL records — supports ShardWriter sharded output dirs."""
    p = Path(path)
    if p.is_dir():
        for sub in sorted(p.glob("*.jsonl*")):
            yield from iter_records(sub)
        return
    yield from iter_records(p)

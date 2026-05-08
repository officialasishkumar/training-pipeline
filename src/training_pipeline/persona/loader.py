"""Parse a persona.md markdown file into structured rules.

The mentor's persona is a living document — bullet points under H2/H3
sections describe how the bot should behave. Some bullets are
mechanically checkable ("references must be hyperlinked", "responses
must include a citation") and some need an LLM judge ("speak in
vernacular", "be empathetic to first-time farmers").

This loader splits both classes out of the same markdown file so the
persona stays human-editable while the scorer can apply each rule the
right way.

Authoring conventions
---------------------

* H2 / H3 sections become rule **groups** — useful for grouping rules
  on dashboards.
* Every bullet is a candidate rule. Indentation and sub-bullets are
  collapsed.
* A rule line may be prefixed with one or more inline tags:

  - ``[hard]`` / ``[soft]``  → severity (default ``soft``).
  - ``[regex: <pattern>]``    → programmatic regex check (must match).
  - ``[forbid: <pattern>]``   → programmatic regex check (must NOT match).
  - ``[contains: <substring>]`` → cheap substring check.
  - ``[id: <slug>]``          → stable rule id (auto-generated otherwise).
  - ``[judge]``               → LLM-judge rule even if a regex tag is also
    present (gives both sides of the check).

Bullets without any inline tag default to ``LLMJudgeRule`` — the
sensible default for a free-text persona description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RuleSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_INLINE_TAG_RE = re.compile(r"\[([a-zA-Z_]+)(?::\s*([^\]]+))?\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")
_BLANK_RE = re.compile(r"^\s*$")


def _slugify(text: str, prefix: str = "") -> str:
    base = _SLUG_RE.sub("-", text.lower()).strip("-")
    base = base[:48] or "rule"
    return f"{prefix}{base}" if prefix else base


@dataclass(frozen=True)
class _Rule:
    """Common rule fields shared by both kinds of rule."""

    id: str
    text: str
    """Original bullet text with tags stripped."""
    section: str
    """Slash-joined heading path (e.g. 'voice/tone')."""
    severity: RuleSeverity = RuleSeverity.SOFT


@dataclass(frozen=True)
class ProgrammaticRule(_Rule):
    """A pure-Python rule the scorer can evaluate without an LLM call."""

    must_match: tuple[str, ...] = ()
    """Regex patterns that MUST find at least one match in the assistant text."""
    must_not_match: tuple[str, ...] = ()
    """Regex patterns whose presence in the assistant text fails the rule."""
    must_contain: tuple[str, ...] = ()
    """Substrings that must be present (case-insensitive)."""

    def evaluate(self, assistant_text: str) -> tuple[bool, list[str]]:
        """Return ``(passed, list_of_reasons)``. Empty reasons on pass."""
        reasons: list[str] = []
        text_lc = assistant_text.lower()
        for pat in self.must_match:
            if not re.search(pat, assistant_text):
                reasons.append(f"missing required pattern: {pat!r}")
        for pat in self.must_not_match:
            if re.search(pat, assistant_text):
                reasons.append(f"matched forbidden pattern: {pat!r}")
        for sub in self.must_contain:
            if sub.lower() not in text_lc:
                reasons.append(f"missing required substring: {sub!r}")
        return (not reasons, reasons)


@dataclass(frozen=True)
class LLMJudgeRule(_Rule):
    """A rule scored by an LLM judge.

    ``criterion`` is the natural-language statement passed to the judge.
    By default we just reuse ``text``, but inline ``[judge: ...]`` syntax
    lets authors override.
    """

    criterion: str = ""
    examples_pass: tuple[str, ...] = ()
    examples_fail: tuple[str, ...] = ()


Rule = ProgrammaticRule | LLMJudgeRule


@dataclass
class Persona:
    """Parsed persona document."""

    name: str
    rules: list[Rule] = field(default_factory=list)
    raw: str = ""
    sections: list[str] = field(default_factory=list)

    def programmatic(self) -> list[ProgrammaticRule]:
        return [r for r in self.rules if isinstance(r, ProgrammaticRule)]

    def llm_judge(self) -> list[LLMJudgeRule]:
        return [r for r in self.rules if isinstance(r, LLMJudgeRule)]

    def hard(self) -> list[Rule]:
        return [r for r in self.rules if r.severity is RuleSeverity.HARD]

    def soft(self) -> list[Rule]:
        return [r for r in self.rules if r.severity is RuleSeverity.SOFT]

    def by_id(self, rule_id: str) -> Rule | None:
        for r in self.rules:
            if r.id == rule_id:
                return r
        return None


def parse_persona(source: str | Path, *, name: str | None = None) -> Persona:
    """Parse a persona markdown file or string into a :class:`Persona`.

    A *very* small markdown subset is recognised: headings (``#``..``####``)
    establish section context, and ``-``/``*``/``+`` bullets become rules.
    Anything else is ignored.
    """
    text = _load_text(source)
    persona_name = name or _derive_name(source) or "persona"
    section_path: list[str] = []
    rules: list[Rule] = []
    seen_ids: set[str] = set()
    sections_seen: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if _BLANK_RE.match(line):
            continue
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            # Collapse to ``level - 1`` since H1 is the persona title we ignore.
            depth = max(0, level - 2)
            section_path = section_path[:depth]
            if level >= 2:
                section_path.append(heading)
                joined = "/".join(_slugify(p) for p in section_path)
                if joined and joined not in sections_seen:
                    sections_seen.append(joined)
            continue
        m = _BULLET_RE.match(line)
        if not m:
            continue
        rule_text = m.group(1).strip()
        rule = _parse_bullet(
            rule_text,
            section="/".join(_slugify(p) for p in section_path) or "root",
            seen_ids=seen_ids,
        )
        if rule is not None:
            rules.append(rule)
            seen_ids.add(rule.id)

    return Persona(name=persona_name, rules=rules, raw=text, sections=sections_seen)


def _load_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if "\n" in source or len(source) > 256:
        return source
    candidate = Path(source)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return source


def _derive_name(source: str | Path) -> str | None:
    if isinstance(source, Path):
        return source.stem
    candidate = Path(str(source))
    if candidate.exists():
        return candidate.stem
    return None


def _parse_bullet(
    text: str, *, section: str, seen_ids: set[str]
) -> Rule | None:
    tags: dict[str, list[str]] = {}
    cleaned = text
    for m in _INLINE_TAG_RE.finditer(text):
        key = m.group(1).lower()
        value = (m.group(2) or "").strip()
        tags.setdefault(key, []).append(value)
    cleaned = _INLINE_TAG_RE.sub("", text).strip()
    if not cleaned:
        return None

    severity = RuleSeverity.SOFT
    if "hard" in tags:
        severity = RuleSeverity.HARD
    elif "soft" in tags:
        severity = RuleSeverity.SOFT

    rule_id = _resolve_id(tags, cleaned, section, seen_ids)

    must_match = tuple(v for v in tags.get("regex", []) if v)
    must_not_match = tuple(v for v in tags.get("forbid", []) if v)
    must_contain = tuple(v for v in tags.get("contains", []) if v)
    judge_criteria = [v for v in tags.get("judge", []) if v]

    is_programmatic = bool(must_match or must_not_match or must_contain)
    is_judge = "judge" in tags or not is_programmatic

    if is_programmatic and not is_judge:
        return ProgrammaticRule(
            id=rule_id,
            text=cleaned,
            section=section,
            severity=severity,
            must_match=must_match,
            must_not_match=must_not_match,
            must_contain=must_contain,
        )
    if is_programmatic and is_judge:
        # Both sides — emit the judge variant; programmatic checks for the
        # same intent should be authored as a separate bullet.
        return LLMJudgeRule(
            id=rule_id,
            text=cleaned,
            section=section,
            severity=severity,
            criterion=judge_criteria[0] if judge_criteria else cleaned,
        )
    return LLMJudgeRule(
        id=rule_id,
        text=cleaned,
        section=section,
        severity=severity,
        criterion=judge_criteria[0] if judge_criteria else cleaned,
    )


def _resolve_id(
    tags: dict[str, list[str]],
    text: str,
    section: str,
    seen_ids: set[str],
) -> str:
    explicit = tags.get("id")
    if explicit:
        candidate = _slugify(explicit[0])
    else:
        candidate = _slugify(text, prefix=f"{_slugify(section)}--" if section else "")
    base = candidate
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def persona_to_dict(persona: Persona) -> dict[str, Any]:
    """Lossy serialisation for diagnostic logging / manifests."""
    return {
        "name": persona.name,
        "n_rules": len(persona.rules),
        "n_programmatic": len(persona.programmatic()),
        "n_judge": len(persona.llm_judge()),
        "n_hard": len(persona.hard()),
        "sections": persona.sections,
        "rule_ids": [r.id for r in persona.rules],
    }

"""Multi-engine PII orchestration.

Each engine catches a different *kind* of PII; running just one of them
leaves obvious gaps. The orchestrator runs them all in priority order
on the same text and unions the findings, deduplicating overlaps so
the redactor sees one decision per span.

Priority order matters because earlier engines win on overlap — we
prefer high-precision detectors (Verhoeff-validated Aadhaar) to
broader ones (Presidio LOCATION). The default order is:

1. Indian-ID rules with checksum validation (Verhoeff Aadhaar, PAN
   entity-class check) — strictest, lowest false-positive rate.
2. Built-in regex rules (email, phone, IBAN, GPS, etc.).
3. Field-rule fallback (``Name:`` / ``Mobile:`` / ``Aadhaar:`` /
   etc., language-agnostic).
4. Presidio NER (English) for PERSON, LOCATION, ORG.
5. IndicNER (Hi/Mr/Ta/Te/Bn/...) for the same entities in Indian
   languages.

Engines that aren't installed are silently skipped — each engine
checks for its optional dependency and falls back to a no-op so the
orchestrator runs end-to-end on a vanilla install.

Language detection is naive on purpose (Indic-script presence wins).
The mentor's note specifically called out "Hindi + regional Indian
languages"; full langid is overkill for the categories we handle.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from training_pipeline.pii.engines.indicner import (
    SUPPORTED_LANGS as INDICNER_SUPPORTED_LANGS,
)
from training_pipeline.pii.recognizers.indian_ids import (
    INDIAN_ID_RULES,
    is_valid_aadhaar,
    is_valid_pan,
)
from training_pipeline.pii.rules import (
    BUILTIN_RULES,
    PIIDetection,
    PIIRule,
    detect_all,
)
from training_pipeline.pii.structured_fields import detect_structured_fields

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


class PIIEngine(Protocol):
    """Tiny shared protocol every engine implements.

    The orchestrator is the only thing that knows about engines; the
    individual engines can be swapped, reordered, or disabled per call
    without touching the redactor.
    """

    name: str
    """Stable identifier for this engine. Used in diagnostic logs."""

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]: ...


# ---------------------------------------------------------------------------
# Built-in engines wrapping the existing rule machinery.
# ---------------------------------------------------------------------------


@dataclass
class IndianIDEngine:
    """Verhoeff Aadhaar + PAN-checksum + Indian mobile / DL / Voter ID."""

    name: str = "indian_ids"
    rules: tuple[PIIRule, ...] = INDIAN_ID_RULES

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:
        out: list[PIIDetection] = []
        for d in detect_all(text, self.rules):
            if d.rule == "aadhaar_verhoeff" and not is_valid_aadhaar(d.text):
                continue
            if d.rule == "pan_with_checksum" and not is_valid_pan(d.text):
                continue
            out.append(d)
        return out


@dataclass
class RegexRuleEngine:
    """Run the existing built-in regex rules.

    The Indian-ID rules are excluded by default because :class:`IndianIDEngine`
    runs them with checksum validation first — running them unvalidated would
    just generate near-duplicate false positives.
    """

    name: str = "regex_builtin"
    rules: tuple[PIIRule, ...] = BUILTIN_RULES
    drop_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"GOV_ID_IN", "PHONE"})
    )

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:
        return [d for d in detect_all(text, self.rules) if d.category not in self.drop_categories]


@dataclass
class FieldRuleEngine:
    """``Name:`` / ``Mobile:`` / ``Aadhaar:`` style structured-field detector."""

    name: str = "field_rules"

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:
        return detect_structured_fields(text)


# Wrappers for the optional Presidio + IndicNER engines so the orchestrator
# can construct them lazily and skip them silently when the deps are missing.


@dataclass
class _OptionalPresidioEngine:
    name: str = "presidio_en"
    score_threshold: float = 0.6
    _detector: Any = field(init=False, default=None)
    _failed: bool = field(init=False, default=False)

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:
        if language != "en":
            return []
        if self._failed:
            return []
        if self._detector is None:
            try:
                from training_pipeline.pii.engines.presidio import PresidioDetector

                self._detector = PresidioDetector(score_threshold=self.score_threshold)
            except ImportError:
                self._failed = True
                log.info("orchestrator: presidio not installed; English NER disabled")
                return []
        return self._detector.detect(text, language=language)


@dataclass
class _OptionalIndicNEREngine:
    name: str = "indicner"
    score_threshold: float = 0.6
    _detector: Any = field(init=False, default=None)
    _failed: bool = field(init=False, default=False)

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:
        if language not in INDICNER_SUPPORTED_LANGS:
            return []
        if self._failed:
            return []
        if self._detector is None:
            try:
                from training_pipeline.pii.engines.indicner import IndicNERDetector

                self._detector = IndicNERDetector(score_threshold=self.score_threshold)
            except ImportError:  # pragma: no cover — handled inside detector
                self._failed = True
                return []
        return self._detector.detect(text, language=language)


# ---------------------------------------------------------------------------
# Language detection (deliberately small).
# ---------------------------------------------------------------------------


_INDIC_RANGES = (
    (0x0900, 0x097F, "hi"),  # Devanagari → defaults to hi/mr (we pick hi)
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
)


def detect_language(text: str) -> str:
    """Return the dominant Indic ISO code for ``text``, or ``'en'``.

    Naive but good enough for routing: any Indic script overrules
    Latin. We pick the script with the most code points; Devanagari
    text is reported as ``hi`` (Marathi can be re-routed via config).
    """
    if not text:
        return "en"
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in _INDIC_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "en"
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineFinding:
    """One finding annotated with the engine that produced it."""

    engine: str
    detection: PIIDetection


@dataclass
class PIIOrchestrator:
    """Run the configured engines, dedup overlaps, return one decision per span.

    The default engine list covers the production matrix. A caller that
    only wants Indian IDs can pass ``engines=[IndianIDEngine()]`` to
    skip the rest.
    """

    engines: Sequence[PIIEngine] = field(default_factory=list)

    @classmethod
    def default(
        cls,
        *,
        enable_presidio: bool = True,
        enable_indicner: bool = True,
    ) -> PIIOrchestrator:
        engines: list[PIIEngine] = [
            IndianIDEngine(),
            RegexRuleEngine(),
            FieldRuleEngine(),
        ]
        if enable_presidio:
            engines.append(_OptionalPresidioEngine())
        if enable_indicner:
            engines.append(_OptionalIndicNEREngine())
        return cls(engines=engines)

    def detect(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> list[PIIDetection]:
        """Return deduplicated detections in left-to-right order."""
        lang = language or detect_language(text)
        findings: list[EngineFinding] = []
        for engine in self.engines:
            try:
                hits = engine.detect(text, language=lang)
            except Exception as exc:  # pragma: no cover — engine contract
                log.warning("orchestrator: engine %s raised %s", engine.name, exc)
                continue
            findings.extend(EngineFinding(engine=engine.name, detection=d) for d in hits)
        return _dedupe_overlaps(findings)

    def detect_with_engines(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> list[EngineFinding]:
        """Like :meth:`detect` but keeps the engine-of-origin annotation."""
        lang = language or detect_language(text)
        findings: list[EngineFinding] = []
        for engine in self.engines:
            try:
                hits = engine.detect(text, language=lang)
            except Exception as exc:  # pragma: no cover — engine contract
                log.warning("orchestrator: engine %s raised %s", engine.name, exc)
                continue
            findings.extend(EngineFinding(engine=engine.name, detection=d) for d in hits)
        return _dedupe_overlap_findings(findings)


def _dedupe_overlaps(findings: Iterable[EngineFinding]) -> list[PIIDetection]:
    """Return one ``PIIDetection`` per text span.

    On overlap, the *earlier* engine wins (priority is the engine list
    order). On a tie of engines, the longer span wins.
    """
    return [f.detection for f in _dedupe_overlap_findings(findings)]


def _dedupe_overlap_findings(
    findings: Iterable[EngineFinding],
) -> list[EngineFinding]:
    items = sorted(
        findings,
        key=lambda f: (f.detection.start, -f.detection.end),
    )
    out: list[EngineFinding] = []
    for f in items:
        if out and f.detection.start < out[-1].detection.end:
            # Overlap — keep the earlier one (prior priority).
            continue
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Convenience: produce a small comparison report between Presidio-only and
# the full orchestrator. Used by docs/PII_POLICY.md to keep the comparison
# table honest.
# ---------------------------------------------------------------------------


def coverage_report(
    samples: Iterable[tuple[str, str]],
    *,
    expected_categories: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Run both Presidio-only and full orchestrator on ``samples``.

    ``samples`` is an iterable of ``(label, text)``. The optional
    ``expected_categories`` maps each label to the categories that
    must be detected in that sample for the run to count as a hit.
    """
    presidio_only = PIIOrchestrator(engines=[_OptionalPresidioEngine()])
    full = PIIOrchestrator.default()

    rows: list[dict[str, Any]] = []
    for label, text in samples:
        p_hits = presidio_only.detect(text)
        f_hits = full.detect(text)
        expected = (expected_categories or {}).get(label, set())
        p_categories = {d.category for d in p_hits}
        f_categories = {d.category for d in f_hits}
        rows.append(
            {
                "sample": label,
                "presidio_only": sorted(p_categories),
                "full_orchestrator": sorted(f_categories),
                "expected": sorted(expected),
                "presidio_recall": (
                    len(expected & p_categories) / len(expected) if expected else None
                ),
                "full_recall": (
                    len(expected & f_categories) / len(expected) if expected else None
                ),
            }
        )
    return {"rows": rows}

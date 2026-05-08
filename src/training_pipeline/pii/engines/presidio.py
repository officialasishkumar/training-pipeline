"""Presidio-based NER detection.

Wraps `presidio-analyzer` if installed. Falls back to a no-op if not, so
the core pipeline stays import-clean without optional ML dependencies.

Use it for entity types that pure regex can't catch — PERSON, LOCATION,
ORGANIZATION. The detector also registers our custom recognizers so
Indian mobiles and Aadhaar/PAN are classified as the right category
(``GOV_ID_IN``/``PHONE``) instead of being misclassified as ``UK_NHS``,
and so callers don't have to add them on every analyze call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import regex as re

from training_pipeline.pii.recognizers.indian_ids import (
    INDIAN_ID_RULES,
    is_valid_aadhaar,
    is_valid_pan,
)
from training_pipeline.pii.rules import PIIDetection

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from presidio_analyzer import AnalyzerEngine


def _build_indian_recognizers() -> list[Any]:  # pragma: no cover — needs presidio
    """Build PatternRecognizer instances for our INDIAN_ID_RULES."""
    from presidio_analyzer import (  # type: ignore[import-not-found]
        Pattern,
        PatternRecognizer,
    )

    recognizers: list[Any] = []
    for rule in INDIAN_ID_RULES:
        pattern = Pattern(name=rule.name, regex=rule.pattern, score=0.85)
        recognizers.append(
            PatternRecognizer(
                supported_entity=rule.category,
                supported_language="en",
                patterns=[pattern],
                name=f"indian_{rule.name}",
            )
        )
    return recognizers


class PresidioDetector:
    """Thin shim around presidio-analyzer.

    Raises ImportError on construction if the optional dependency is missing.
    Custom Indian recognizers are added to the registry so callers don't
    have to register them on every analyze call.
    """

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("en",),
        entities: tuple[str, ...] = (
            "PERSON",
            "LOCATION",
            "ORGANIZATION",
            "DATE_TIME",
            "PHONE",
            "GOV_ID_IN",
        ),
        score_threshold: float = 0.6,
        register_indian_recognizers: bool = True,
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "presidio-analyzer is not installed. "
                'Install training-pipeline with the "ner" extra: '
                'pip install "training-pipeline[ner]"'
            ) from exc

        self._engine: AnalyzerEngine = AnalyzerEngine()
        if register_indian_recognizers:  # pragma: no cover — needs presidio
            for rec in _build_indian_recognizers():
                self._engine.registry.add_recognizer(rec)
        self.languages = languages
        self.entities = entities
        self.score_threshold = score_threshold

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:  # pragma: no cover
        results = self._engine.analyze(
            text=text,
            entities=list(self.entities),
            language=language,
            score_threshold=self.score_threshold,
        )
        out: list[PIIDetection] = []
        for r in results:
            span = text[r.start : r.end]
            # Validate Aadhaar/PAN spans before accepting them. Bare regex matches
            # generate too many false positives in farmer-profile context.
            if r.entity_type == "GOV_ID_IN":
                upper = span.upper()
                if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", upper):
                    if not is_valid_pan(span):
                        continue
                elif re.fullmatch(r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}", span) and not is_valid_aadhaar(span):
                    continue
            out.append(
                PIIDetection(
                    rule=f"presidio_{r.entity_type.lower()}",
                    category=r.entity_type,
                    start=r.start,
                    end=r.end,
                    text=span,
                )
            )
        return out

"""Optional ML-assisted PII detection.

Wraps `presidio-analyzer` if installed. Falls back to a no-op if not, so the
core pipeline stays import-clean without optional ML dependencies.

Use it for entity types that pure regex can't catch — PERSON, LOCATION,
ORGANIZATION. Combine with the rule-based detector for best coverage:

.. code-block:: python

    from training_pipeline.pii.rules import BUILTIN_RULES, detect_all
    from training_pipeline.pii.ner import PresidioDetector

    presidio = PresidioDetector()
    rule_dets = detect_all(text, BUILTIN_RULES)
    ner_dets = presidio.detect(text)
    all_dets = sorted(rule_dets + ner_dets, key=lambda d: (d.start, -d.end))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from training_pipeline.pii.rules import PIIDetection

if TYPE_CHECKING:  # pragma: no cover
    from presidio_analyzer import AnalyzerEngine


class PresidioDetector:
    """Thin shim around presidio-analyzer.

    Raises ImportError on construction if the optional dependency is missing.
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
        ),
        score_threshold: float = 0.6,
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "presidio-analyzer is not installed. "
                'Install training-pipeline with the "ner" extra: '
                'pip install "training-pipeline[ner]"'
            ) from exc

        self._engine: AnalyzerEngine = AnalyzerEngine()
        self.languages = languages
        self.entities = entities
        self.score_threshold = score_threshold

    def detect(self, text: str, *, language: str = "en") -> list[PIIDetection]:
        results = self._engine.analyze(
            text=text,
            entities=list(self.entities),
            language=language,
            score_threshold=self.score_threshold,
        )
        return [
            PIIDetection(
                rule=f"presidio_{r.entity_type.lower()}",
                category=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start : r.end],
            )
            for r in results
        ]

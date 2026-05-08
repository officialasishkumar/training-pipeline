"""IndicNER wrapper for multilingual PII (PERSON, LOCATION, ORG).

Presidio's English NER misses Indian names and Hindi/Marathi/Tamil
locations entirely. ``ai4bharat/IndicNER`` (and successors) is the
standard way to fill that gap. We treat it as an *optional* detector
under the ``[indic]`` extra so the base install is still a pure-Python
package.

The wrapper is intentionally narrow — given text and a language hint,
return the same ``PIIDetection`` shape every other engine returns. The
orchestrator handles unioning, dedup, and placeholder assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from training_pipeline.pii.rules import PIIDetection

log = logging.getLogger(__name__)


# ai4bharat/IndicNER ships fine-tunes for these languages; we accept the
# ISO-639-1 codes the rest of the pipeline uses.
SUPPORTED_LANGS: tuple[str, ...] = (
    "hi",  # Hindi
    "mr",  # Marathi
    "ta",  # Tamil
    "te",  # Telugu
    "bn",  # Bengali
    "gu",  # Gujarati
    "kn",  # Kannada
    "ml",  # Malayalam
    "or",  # Oriya
    "pa",  # Punjabi
    "as",  # Assamese
)


# IndicNER label set → our PIIDetection categories.
_LABEL_TO_CATEGORY = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "B-PER": "PERSON",
    "I-PER": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "B-LOC": "LOCATION",
    "I-LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "B-ORG": "ORGANIZATION",
    "I-ORG": "ORGANIZATION",
}


@dataclass
class IndicNERDetector:
    """Lazy wrapper around an IndicNER token-classification pipeline.

    The model is loaded the first time ``detect`` is called so a process
    that never sees Indic text doesn't pay the load cost. A no-op
    fallback is used if the optional dependency isn't installed and
    ``strict=False`` (the default).
    """

    model_id: str = "ai4bharat/IndicNER"
    aggregation_strategy: str = "simple"
    """Passed through to the HF token-classification pipeline."""
    score_threshold: float = 0.6
    strict: bool = False
    """When true, raise ImportError if the optional dep is missing.
    When false (default), log a warning and return no detections."""

    _pipeline: Any = field(init=False, default=None)
    _load_failed: bool = field(init=False, default=False)

    def _load(self) -> None:
        if self._pipeline is not None or self._load_failed:
            return
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            self._load_failed = True
            if self.strict:
                raise ImportError(
                    "transformers is not installed. "
                    'Install training-pipeline with the "indic" extra: '
                    'pip install "training-pipeline[indic]"'
                ) from exc
            log.warning(
                "IndicNERDetector: transformers not installed; "
                "Indic NER will be a no-op."
            )
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForTokenClassification.from_pretrained(self.model_id)
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy=self.aggregation_strategy,
            )
        except Exception as exc:  # pragma: no cover — model-load runtime
            self._load_failed = True
            if self.strict:
                raise
            log.warning("IndicNERDetector: model load failed (%s); falling back to no-op.", exc)

    def detect(self, text: str, *, language: str = "hi") -> list[PIIDetection]:
        """Return PERSON / LOCATION / ORGANIZATION spans found in ``text``.

        Languages outside :data:`SUPPORTED_LANGS` short-circuit to an
        empty list — the orchestrator should route those to other
        engines (Presidio for English, regex for IDs).
        """
        if language not in SUPPORTED_LANGS:
            return []
        self._load()
        if self._pipeline is None:
            return []
        try:
            raw = self._pipeline(text)
        except Exception as exc:  # pragma: no cover — runtime model failure
            log.warning("IndicNER inference failed: %s", exc)
            return []

        out: list[PIIDetection] = []
        for ent in raw:
            score = float(ent.get("score", 0.0))
            if score < self.score_threshold:
                continue
            label = str(ent.get("entity_group") or ent.get("entity") or "").upper()
            category = _LABEL_TO_CATEGORY.get(label)
            if not category:
                continue
            start = int(ent.get("start", 0))
            end = int(ent.get("end", start))
            if start >= end or end > len(text):
                continue
            out.append(
                PIIDetection(
                    rule=f"indicner_{category.lower()}",
                    category=category,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
        return out

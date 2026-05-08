"""PII detection engines.

Each engine has a ``detect(text, *, language='en') -> list[PIIDetection]``
contract; the orchestrator runs them in priority order and unions the
results. Engines that depend on heavy ML libraries (Presidio, IndicNER)
import them lazily so the base install stays light.
"""

from training_pipeline.pii.engines.indicner import (
    SUPPORTED_LANGS as INDICNER_SUPPORTED_LANGS,
)
from training_pipeline.pii.engines.indicner import (
    IndicNERDetector,
)

__all__ = ["INDICNER_SUPPORTED_LANGS", "IndicNERDetector"]

"""Back-compat shim — :class:`PresidioDetector` moved to ``pii.engines.presidio``.

Existing code importing ``training_pipeline.pii.ner.PresidioDetector`` keeps
working. New code should import from ``training_pipeline.pii.engines``.
"""

from training_pipeline.pii.engines.presidio import PresidioDetector

__all__ = ["PresidioDetector"]

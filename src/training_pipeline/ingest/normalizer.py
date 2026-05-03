"""Normalize raw log records into canonical Trajectories.

The normalizer is the single entry point used by ``tp ingest``. It picks the
right source adapter (auto-detect or forced), runs it, and yields validated
Trajectories along with any error captured during conversion.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from training_pipeline.ingest.sources import detect_source, get_source
from training_pipeline.schemas.events import Trajectory

log = logging.getLogger(__name__)


@dataclass
class NormalizationError:
    """Captured failure during a single record conversion.

    Surfaced by ``normalize_records`` so the caller can write a side-channel of
    rejected records for human review without aborting the run.
    """

    index: int
    error: str
    record: dict[str, Any]


def normalize_record(
    record: dict[str, Any], *, source: str | None = None
) -> Trajectory:
    """Convert one raw record to a Trajectory.

    If ``source`` is ``None`` the adapter is sniffed from the record shape.
    """
    name = source or detect_source(record)
    adapter = get_source(name)
    return adapter(record)


def normalize_records(
    records: Iterable[dict[str, Any]], *, source: str | None = None
) -> Iterator[Trajectory | NormalizationError]:
    """Stream normalize an iterable of records.

    Errors are yielded inline as ``NormalizationError`` so the caller can decide
    whether to abort, log, or write them to a quarantine file.
    """
    for i, rec in enumerate(records):
        try:
            yield normalize_record(rec, source=source)
        except Exception as exc:  # noqa: BLE001 - we surface every failure
            log.warning("normalize failed at record %d: %s", i, exc)
            yield NormalizationError(index=i, error=str(exc), record=rec)


def normalize_session(records: Iterable[dict[str, Any]], *, source: str | None = None) -> Trajectory:
    """Convenience: normalize a single record and return the trajectory.

    Raises if normalization fails.
    """
    items = list(records)
    if len(items) != 1:
        raise ValueError(f"normalize_session expects exactly one record, got {len(items)}")
    return normalize_record(items[0], source=source)

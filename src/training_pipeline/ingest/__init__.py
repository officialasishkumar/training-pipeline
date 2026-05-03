"""Ingest layer: read heterogeneous logs into the canonical schema."""

from training_pipeline.ingest.normalizer import normalize_record, normalize_session
from training_pipeline.ingest.parsers import (
    iter_jsonl,
    iter_records,
    write_jsonl,
)
from training_pipeline.ingest.sources import detect_source, register_source

__all__ = [
    "detect_source",
    "iter_jsonl",
    "iter_records",
    "normalize_record",
    "normalize_session",
    "register_source",
    "write_jsonl",
]

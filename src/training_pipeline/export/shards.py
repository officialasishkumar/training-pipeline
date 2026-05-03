"""Sharded JSONL writer with dataset-card metadata.

Trainers running across many GPUs prefer fixed-size shards (and the HF
``datasets`` library auto-discovers them). This writer rotates files at a
configurable record count and emits a JSON ``dataset_card.json`` summarising
shape, fingerprint, and stratification breakdown.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import orjson

log = logging.getLogger(__name__)


class ShardWriter:
    """Round-robin JSONL writer that rotates files at ``shard_size`` rows.

    Usage:

    .. code-block:: python

        with ShardWriter("build/sft", shard_size=5000, prefix="train") as w:
            for record in records:
                w.write(record)
        # produces build/sft/train-00000.jsonl, train-00001.jsonl, ...
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_size: int = 5000,
        prefix: str = "shard",
        suffix: str = ".jsonl",
        compress: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, shard_size)
        self.prefix = prefix
        self.suffix = suffix + (".gz" if compress else "")
        self.compress = compress
        self._shard_idx = 0
        self._row_idx = 0
        self._fh: Any = None
        self._fingerprint = hashlib.sha256()
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self.output_dir / f"{self.prefix}-{self._shard_idx:05d}{self.suffix}"
        self._fh = gzip.open(path, "ab") if self.compress else path.open("ab")
        self._row_idx = 0

    def write(self, record: Any) -> None:
        if hasattr(record, "model_dump"):
            payload = record.model_dump(mode="json", exclude_none=True)
        else:
            payload = record
        line = orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
        self._fh.write(line)
        self._fingerprint.update(line)
        self._row_idx += 1
        if self._row_idx >= self.shard_size:
            self._shard_idx += 1
            self._open_new_shard()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()

    @property
    def shard_count(self) -> int:
        return self._shard_idx + (1 if self._row_idx > 0 else 0)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def write_dataset_card(
    output_dir: str | Path,
    *,
    name: str,
    record_count: int,
    fingerprint: str,
    fields: list[str],
    chat_template: str | None = None,
    pii_report: dict[str, int] | None = None,
    strata: dict[str, dict[str, int]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a ``dataset_card.json`` describing the export.

    The card is intentionally JSON (not YAML) so loaders can ingest it without
    extra deps, and includes a fingerprint so downstream pipelines can detect
    silent dataset changes.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    card = {
        "name": name,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "record_count": record_count,
        "fingerprint": fingerprint,
        "fields": fields,
        "chat_template": chat_template,
        "pii_report": pii_report or {},
        "strata": strata or {},
        "training_pipeline_version": _our_version(),
        **(extra or {}),
    }
    path = out / "dataset_card.json"
    path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _our_version() -> str:
    try:
        from training_pipeline import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"

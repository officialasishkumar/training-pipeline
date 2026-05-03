"""Streaming JSONL parsers tolerant of real-world log quirks.

The pipeline never loads a full corpus into memory — every entry point yields
records lazily so a 50 GB log file is just as cheap as a 50 KB one.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

import orjson


def _open_text(path: Path) -> TextIO:
    """Open text or .gz transparently."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")  # type: ignore[return-value]
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: str | os.PathLike[str], *, strict: bool = False) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-blank line.

    Logs sometimes contain blank lines and trailing junk — by default we skip
    silently and continue. Pass ``strict=True`` to raise instead.
    """
    p = Path(path)
    with _open_text(p) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"{p}:{line_no} invalid JSON: {exc}") from exc
                continue


def iter_records(
    path: str | os.PathLike[str], *, strict: bool = False
) -> Iterator[dict[str, Any]]:
    """Yield records from a path that is either a file or a directory.

    Directories are walked recursively; both ``.jsonl`` and ``.jsonl.gz``
    files are picked up. Plain ``.json`` files containing an array of records
    are also supported.
    """
    p = Path(path)
    if p.is_file():
        yield from _iter_file(p, strict=strict)
        return
    if p.is_dir():
        for sub in sorted(p.rglob("*")):
            if sub.is_file() and sub.name.endswith((".jsonl", ".jsonl.gz", ".json")):
                yield from _iter_file(sub, strict=strict)
        return
    raise FileNotFoundError(f"No such file or directory: {path}")


def _iter_file(path: Path, *, strict: bool) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".json") and not path.name.endswith(".jsonl"):
        with _open_text(path) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            yield from (r for r in data if isinstance(r, dict))
        elif isinstance(data, dict):
            yield data
        return
    yield from iter_jsonl(path, strict=strict)


def write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[Any],
    *,
    append: bool = False,
) -> int:
    """Write an iterable of records (dicts or Pydantic models) to JSONL.

    Returns the number of records written. Parent directories are created.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if append else "wb"
    count = 0
    if p.suffix == ".gz":
        opener = gzip.open(p, mode)
    else:
        opener = p.open(mode)
    with opener as fh:
        for record in records:
            payload = record
            if hasattr(record, "model_dump"):
                payload = record.model_dump(mode="json", exclude_none=False)
            line = orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
            fh.write(line)
            count += 1
    return count

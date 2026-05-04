"""Run manifests: reproducibility hashes and audit trails for pipeline runs.

A *manifest* records exactly which inputs produced which outputs under which
config and which version of the pipeline code, so a later reviewer can:

- recreate a published dataset bit-for-bit (with the same input + same code),
- detect silent regressions (config changed but outputs are stamped with the
  same name),
- audit which raw log lines contributed to a trained model.

This module is deliberately small and dependency-free: it walks files,
hashes them with SHA-256 in fixed-size chunks, and writes a JSON manifest.
``write_manifest`` is the high-level entry point used by the CLI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def hash_file(path: str | Path, *, algorithm: str = "sha256") -> str:
    """Stream a file into a hash and return the hex digest."""
    h = hashlib.new(algorithm)
    p = Path(path)
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes, *, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


def hash_obj(obj: Any, *, algorithm: str = "sha256") -> str:
    """Stable hash of any JSON-serialisable object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hash_bytes(payload, algorithm=algorithm)


class FileEntry(BaseModel):
    """One file recorded in a manifest."""

    model_config = ConfigDict(extra="forbid")

    path: str
    bytes: int
    sha256: str
    role: str = Field(
        default="output",
        description="'input' | 'output' | 'config' | 'shard' | 'audit' — free-form label.",
    )


class StageEntry(BaseModel):
    """One pipeline stage's contribution to the manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str
    started_at: datetime
    finished_at: datetime
    files: list[FileEntry] = Field(default_factory=list)
    counters: dict[str, int] = Field(default_factory=dict)
    notes: str | None = None


class RunManifest(BaseModel):
    """Top-level manifest for a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    pipeline_version: str
    created_at: datetime
    config_hash: str | None = None
    config_snapshot: dict[str, Any] | None = None
    stages: list[StageEntry] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


def file_entries(
    paths: Iterable[str | Path],
    *,
    role: str = "output",
    base_dir: Path | None = None,
) -> list[FileEntry]:
    """Build :class:`FileEntry` objects for a list of files.

    ``base_dir``, when given, is used to record paths *relative* to it so
    manifests stay portable when a build is moved across directories. Both
    the source paths and the base are resolved to absolute first so a mix of
    relative and absolute inputs works the same way.
    """
    out: list[FileEntry] = []
    base_abs = base_dir.resolve() if base_dir else None
    for raw in paths:
        p = Path(raw)
        recorded = (
            str(p.resolve().relative_to(base_abs)) if base_abs is not None else str(p)
        )
        out.append(
            FileEntry(
                path=recorded,
                bytes=p.stat().st_size,
                sha256=hash_file(p),
                role=role,
            )
        )
    return out


def discover_files(
    root: str | Path,
    *,
    patterns: tuple[str, ...] = ("*.jsonl", "*.jsonl.gz", "*.json"),
    role: str = "output",
    base_dir: str | Path | None = None,
) -> list[FileEntry]:
    """Glob ``root`` and build entries for matching files. Useful for shard dirs.

    ``base_dir`` controls how paths are recorded. When ``None`` (the default),
    paths are recorded relative to ``root``. Pass an explicit anchor (e.g. the
    pipeline cwd) when several stages share one manifest so all entries use
    the same path frame and ``verify_manifest`` can resolve them with a single
    base.
    """
    root_path = Path(root)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root_path.rglob(pattern)))
    # Stable order, no duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    anchor = Path(base_dir) if base_dir is not None else root_path
    return file_entries(unique, role=role, base_dir=anchor)


def make_run_id(config_hash: str, started_at: datetime) -> str:
    """Compact run id combining time and config hash for sortable uniqueness."""
    ts = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"run-{ts}-{config_hash[:8]}"


def write_manifest(
    path: str | Path,
    manifest: RunManifest,
) -> Path:
    """Serialise a :class:`RunManifest` as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return p


def load_manifest(path: str | Path) -> RunManifest:
    return RunManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest: RunManifest, *, base_dir: str | Path) -> list[str]:
    """Recompute every file hash; return a list of mismatch messages.

    Empty list = manifest matches the on-disk state exactly. Useful as an
    integrity check before downstream consumers (training jobs, releases)
    pick up a build.
    """
    base = Path(base_dir)
    errors: list[str] = []
    for stage in manifest.stages:
        for entry in stage.files:
            target = base / entry.path
            if not target.exists():
                errors.append(f"missing: {entry.path}")
                continue
            actual = hash_file(target)
            if actual != entry.sha256:
                errors.append(
                    f"hash mismatch: {entry.path} expected {entry.sha256[:8]} got {actual[:8]}"
                )
    return errors


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

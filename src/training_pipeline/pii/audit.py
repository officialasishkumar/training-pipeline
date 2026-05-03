"""Audit sampling for human review of PII redaction.

Every release of a training artifact is supposed to ship with a documented
audit sample. This module gives a small deterministic sampler so the same
seed produces the same sample on rerun (re-investigable).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class AuditSampler:
    """Reservoir-sampling helper.

    Uses a hash-of-id per record so the audit set is *stable* given the same
    seed and inputs — important so reviewers don't have to re-look at the
    same data each rerun.
    """

    rate: float = 0.05
    seed: int = 0
    cap: int = 1000

    def __post_init__(self) -> None:
        if not 0 <= self.rate <= 1:
            raise ValueError("rate must be in [0, 1]")
        self._reservoir: list[dict[str, Any]] = []

    def consider(self, record: dict[str, Any], *, key: str | None = None) -> bool:
        """Decide whether to keep this record. Returns True if kept."""
        if not record:
            return False
        ident = key or self._derive_key(record)
        h = hashlib.sha256(f"{self.seed}:{ident}".encode()).digest()
        bucket = int.from_bytes(h[:8], "big") / 2**64
        if bucket >= self.rate:
            return False
        if len(self._reservoir) >= self.cap:
            # Replace using deterministic index based on hash so it's still stable.
            idx = int.from_bytes(h[8:16], "big") % self.cap
            self._reservoir[idx] = record
        else:
            self._reservoir.append(record)
        return True

    def _derive_key(self, record: dict[str, Any]) -> str:
        return str(
            record.get("session_id")
            or record.get("event_id")
            or record.get("id")
            or random.random()
        )

    def consume(self) -> list[dict[str, Any]]:
        """Return the collected sample and clear the reservoir."""
        out, self._reservoir = self._reservoir, []
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._reservoir)

    def __len__(self) -> int:
        return len(self._reservoir)


def write_audit_summary(
    samples: Iterable[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, Any]:
    """Build a dict suitable for a JSON summary alongside the audit JSONL."""
    samples_list = list(samples)
    return {
        "audit_total": len(samples_list),
        "category_counts": dict(sorted(counts.items())),
        "categories_seen": sorted(counts),
        "version": 1,
    }

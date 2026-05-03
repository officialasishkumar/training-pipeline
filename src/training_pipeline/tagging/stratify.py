"""Stratified sampling and split utilities for diverse training mixtures.

Two functions:

- ``stratum_key`` derives a stable key from trajectory tags (complexity_band,
  domain, optional bucketed tool_set) so callers can stratify any way they
  like by changing the keys list.
- ``stratified_split`` produces train/val/test indices that approximately
  balance the strata while staying deterministic given a seed.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from training_pipeline.schemas.events import Trajectory


@dataclass
class StratifiedSplit:
    train: list[int]
    val: list[int]
    test: list[int]
    strata: dict[str, dict[str, int]]
    """Per-stratum count breakdown: {key: {'train': n, 'val': n, 'test': n}}."""


def stratum_key(
    trajectory: Trajectory,
    *,
    keys: Sequence[str] = ("complexity_band", "domain"),
    bucket_tools: bool = False,
) -> str:
    """Build a stratification key from trajectory tags."""
    parts: list[str] = []
    complexity = trajectory.tags.get("complexity") if isinstance(trajectory.tags, dict) else None
    for k in keys:
        if k == "complexity_band":
            parts.append(str((complexity or {}).get("complexity_band", "unknown")))
        elif k == "domain":
            parts.append(str(trajectory.domain or "unknown"))
        elif k == "tool_arity":
            n = (complexity or {}).get("distinct_tools", 0)
            parts.append("0" if n == 0 else "1" if n == 1 else "2+")
        else:
            parts.append(str(trajectory.tags.get(k, "unknown")))
    if bucket_tools:
        tools = (complexity or {}).get("tool_set") or trajectory.tool_set()
        parts.append("|".join(sorted(tools)) or "no_tools")
    return "::".join(parts)


def stratified_split(
    trajectories: Sequence[Trajectory],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    keys: Sequence[str] = ("complexity_band", "domain"),
) -> StratifiedSplit:
    """Return a stratified train/val/test split.

    The split is deterministic for a given (seed, trajectories ordering, keys).
    Each stratum is shuffled with a key-derived seed so adding new strata
    doesn't reshuffle existing ones.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")
    if any(f < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")

    by_stratum: dict[str, list[int]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        by_stratum[stratum_key(traj, keys=keys)].append(i)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    breakdown: dict[str, dict[str, int]] = {}

    for key, indices in by_stratum.items():
        rng = random.Random(_stable_seed(seed, key))
        local = list(indices)
        rng.shuffle(local)
        n = len(local)
        n_train = int(round(n * fractions[0]))
        n_val = int(round(n * fractions[1]))
        # Anything left goes to test so totals add up exactly.
        n_test = max(0, n - n_train - n_val)
        train.extend(local[:n_train])
        val.extend(local[n_train : n_train + n_val])
        test.extend(local[n_train + n_val : n_train + n_val + n_test])
        breakdown[key] = {"train": n_train, "val": n_val, "test": n_test}

    train.sort()
    val.sort()
    test.sort()
    return StratifiedSplit(train=train, val=val, test=test, strata=breakdown)


def _stable_seed(seed: int, key: str) -> int:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big")

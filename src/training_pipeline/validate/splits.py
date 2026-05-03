"""Split-integrity checks: prevent train/val/test (and SFT/DPO) leakage.

Uses MinHash + LSH for approximate near-duplicate detection at scale. We
shingle on character 5-grams of user-turn text — empirical sweet spot for
catching minor rewordings without false-positive overload.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from training_pipeline.schemas.events import Trajectory, UserEvent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DuplicateLeak:
    side_a: str  # split or dataset name
    id_a: int
    side_b: str
    id_b: int
    jaccard: float


def _user_text(traj: Trajectory) -> str:
    return " ".join(e.content for e in traj.events if isinstance(e, UserEvent)).strip()


def _shingles(text: str, *, k: int = 5) -> set[str]:
    text = text.lower()
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def detect_near_duplicates(
    trajectories_a: Sequence[Trajectory],
    trajectories_b: Sequence[Trajectory] | None = None,
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    label_a: str = "a",
    label_b: str = "b",
) -> list[DuplicateLeak]:
    """Return likely duplicates across (or within) the supplied lists.

    If ``trajectories_b`` is None, checks for duplicates *inside*
    ``trajectories_a`` (e.g. within a single split).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "datasketch is required for near-duplicate detection. "
            "Install it with `pip install datasketch`."
        ) from exc

    same_set = trajectories_b is None
    items_b = list(trajectories_b) if trajectories_b is not None else list(trajectories_a)
    items_a = list(trajectories_a)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs_b: list[MinHash] = []
    shingles_b: list[set[str]] = []
    for i, traj in enumerate(items_b):
        text = _user_text(traj)
        sh = _shingles(text)
        m = MinHash(num_perm=num_perm)
        for s in sh:
            m.update(s.encode("utf-8"))
        sigs_b.append(m)
        shingles_b.append(sh)
        lsh.insert(f"b:{i}", m)

    leaks: list[DuplicateLeak] = []
    for i, traj in enumerate(items_a):
        text = _user_text(traj)
        sh = _shingles(text)
        m = MinHash(num_perm=num_perm)
        for s in sh:
            m.update(s.encode("utf-8"))
        for cand in lsh.query(m):
            j = int(cand.split(":", 1)[1])
            if same_set and j <= i:
                continue
            jacc = _jaccard(sh, shingles_b[j])
            if jacc >= threshold:
                leaks.append(
                    DuplicateLeak(
                        side_a=label_a,
                        id_a=i,
                        side_b=label_a if same_set else label_b,
                        id_b=j,
                        jaccard=jacc,
                    )
                )
    return leaks


def split_integrity_report(
    splits: dict[str, Sequence[Trajectory]],
    *,
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Pairwise leak check across named splits.

    Args:
        splits: mapping from split name (e.g. ``"train"``, ``"val"``, ``"test"``,
            ``"dpo"``) to a list of trajectories.
        threshold: Jaccard similarity above which to flag a leak.

    Returns a structured report ``{"leaks_by_pair": {...}, "total_leaks": N}``.
    """
    leaks_by_pair: dict[str, list[dict[str, Any]]] = {}
    total = 0
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            leaks = detect_near_duplicates(
                splits[a],
                splits[b],
                threshold=threshold,
                label_a=a,
                label_b=b,
            )
            if leaks:
                leaks_by_pair[f"{a}<->{b}"] = [
                    {
                        "split_a": leak.side_a,
                        "id_a": leak.id_a,
                        "split_b": leak.side_b,
                        "id_b": leak.id_b,
                        "jaccard": round(leak.jaccard, 3),
                    }
                    for leak in leaks
                ]
                total += len(leaks)
    return {
        "threshold": threshold,
        "total_leaks": total,
        "leaks_by_pair": leaks_by_pair,
    }

"""Cluster user questions across canonical logs to build a diverse seed pool.

Production agentic logs are heavily long-tailed: the same intent shows up
hundreds of times with cosmetic variations (different phrasing, language,
typos). Behaviour-cloning on raw traces over-trains on the head and starves
the tail. ``SeedExtractor`` collapses the head — embed every user question,
cluster, keep one representative per cluster plus the cluster size — so the
synthetic generator is driven by a *broader* set of intents than the raw
logs would imply.

The class supports two embedder backends (``sentence-transformers`` for real
quality; a hash-based char-ngram fallback for tests and offline runs) and
two clustering backends (sklearn ``KMeans`` for sized partitions; a greedy
threshold-based grouping that always works without sklearn).

Output rows include ``cluster_size`` (so the downstream generator can
oversample under-represented intents) and ``original_lineage_ids`` so a
seed can always be traced back to the source traces it summarises.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from training_pipeline.ingest.parsers import iter_records, write_jsonl
from training_pipeline.schemas.events import Trajectory, UserEvent

log = logging.getLogger(__name__)

EmbedderName = Literal["sentence-transformers", "hash"]
ClusterMethod = Literal["kmeans", "greedy"]

_DEFAULT_HASH_DIM = 256
_DEFAULT_NGRAM_RANGE = (3, 5)


@dataclass
class Seed:
    """One representative query for a cluster of similar log questions."""

    seed_id: str
    query: str
    cluster_id: int
    cluster_size: int
    original_lineage_ids: list[str] = field(default_factory=list)
    domain: str | None = None
    source_session_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "query": self.query,
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "original_lineage_ids": list(self.original_lineage_ids),
            "domain": self.domain,
            "source_session_ids": list(self.source_session_ids),
        }


def _first_user_question(traj: Trajectory) -> str | None:
    for ev in traj.events:
        if isinstance(ev, UserEvent):
            text = ev.content.strip()
            if text:
                return text
    return None


def _hash_embed(text: str, dim: int = _DEFAULT_HASH_DIM) -> list[float]:
    """Char-ngram hashing trick. Always available, deterministic, no deps.

    Quality is well below a sentence transformer but it's good enough to
    cluster textually similar queries — and crucially makes the test suite
    runnable on a vanilla install.
    """
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if not cleaned:
        return [0.0] * dim
    vec = [0.0] * dim
    n_lo, n_hi = _DEFAULT_NGRAM_RANGE
    for n in range(n_lo, n_hi + 1):
        for i in range(len(cleaned) - n + 1):
            ng = cleaned[i : i + n]
            h = int(hashlib.blake2b(ng.encode("utf-8"), digest_size=4).hexdigest(), 16)
            sign = 1.0 if (h & 1) else -1.0
            vec[h % dim] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _embed_sentence_transformers(texts: Sequence[str], model_id: str) -> list[list[float]]:
    """Lazy wrapper over sentence-transformers. Falls back to ``_hash_embed``
    if the optional dep is missing — caller logs a warning."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with `pip install training-pipeline[generate]` "
            "or pass embedder='hash'."
        ) from exc
    model = SentenceTransformer(model_id)
    embeds = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in embeds]


def _kmeans_cluster(vectors: Sequence[Sequence[float]], k: int, seed: int) -> list[int]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        from sklearn.cluster import KMeans  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install with `pip install training-pipeline[generate]` "
            "or pass cluster_method='greedy'."
        ) from exc
    if not vectors:
        return []
    arr = np.asarray(vectors, dtype="float32")
    k = max(1, min(k, len(vectors)))
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return [int(x) for x in km.fit_predict(arr)]


def _greedy_cluster(
    vectors: Sequence[Sequence[float]], threshold: float
) -> list[int]:
    """Single-pass greedy clustering. Each item joins the first existing
    cluster whose centroid (running mean) cosine >= threshold; otherwise
    it starts a new cluster.

    Output preserves input order; cluster ids are assigned in first-seen
    order. Deterministic given input order.
    """
    if not vectors:
        return []
    centroids: list[list[float]] = []
    counts: list[int] = []
    labels: list[int] = []
    for v in vectors:
        best_idx = -1
        best_sim = -1.0
        for j, c in enumerate(centroids):
            sim = _cosine(v, c)
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_idx >= 0 and best_sim >= threshold:
            cnt = counts[best_idx]
            centroids[best_idx] = [
                (centroids[best_idx][i] * cnt + v[i]) / (cnt + 1) for i in range(len(v))
            ]
            # Re-normalise to keep cosine comparisons cheap.
            norm = math.sqrt(sum(x * x for x in centroids[best_idx])) or 1.0
            centroids[best_idx] = [x / norm for x in centroids[best_idx]]
            counts[best_idx] += 1
            labels.append(best_idx)
        else:
            centroids.append(list(v))
            counts.append(1)
            labels.append(len(centroids) - 1)
    return labels


def _pick_representative(
    texts: Sequence[str], vectors: Sequence[Sequence[float]], indices: Sequence[int]
) -> int:
    """Return the index in ``indices`` of the centroid-nearest item.

    On ties (or single-element clusters) returns the shortest, lexicographically
    earliest text — keeps output deterministic.
    """
    if len(indices) == 1:
        return indices[0]
    dim = len(vectors[indices[0]])
    centroid = [0.0] * dim
    for i in indices:
        for d in range(dim):
            centroid[d] += vectors[i][d]
    centroid = [c / len(indices) for c in centroid]
    norm = math.sqrt(sum(c * c for c in centroid)) or 1.0
    centroid = [c / norm for c in centroid]
    best = indices[0]
    best_score: tuple[float, int, str] | None = None
    for i in indices:
        score = _cosine(vectors[i], centroid)
        # Sort key: higher cosine wins; tie-break on shorter text then lex.
        key = (-score, len(texts[i]), texts[i])
        if best_score is None or key < best_score:
            best_score = key
            best = i
    return best


@dataclass
class SeedExtractor:
    """Cluster user questions across trajectories and emit one seed per cluster."""

    embedder: EmbedderName = "hash"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cluster_method: ClusterMethod = "greedy"
    n_clusters: int | None = None
    """For ``kmeans``: target cluster count. Auto-sized when None."""
    similarity_threshold: float = 0.72
    """For ``greedy``: minimum cosine to join an existing cluster."""
    seed: int = 0
    min_query_chars: int = 3

    def extract(self, trajectories: Iterable[Trajectory]) -> list[Seed]:
        items = self._collect(trajectories)
        if not items:
            return []
        texts = [it["query"] for it in items]
        vectors = self._embed(texts)
        labels = self._cluster(vectors)
        return self._build_seeds(items, texts, vectors, labels)

    def extract_to_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> int:
        trajectories = (Trajectory.model_validate(r) for r in iter_records(input_path))
        seeds = self.extract(trajectories)
        write_jsonl(output_path, (s.as_dict() for s in seeds))
        return len(seeds)

    def _collect(self, trajectories: Iterable[Trajectory]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for traj in trajectories:
            q = _first_user_question(traj)
            if not q or len(q) < self.min_query_chars:
                continue
            items.append(
                {
                    "query": q,
                    "domain": traj.domain,
                    "session_id": traj.session_id,
                    "lineage_id": traj.lineage_id,
                }
            )
        return items

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.embedder == "sentence-transformers":
            try:
                return _embed_sentence_transformers(texts, self.embedder_model)
            except ImportError as exc:
                log.warning("falling back to hash embedder: %s", exc)
        return [_hash_embed(t) for t in texts]

    def _cluster(self, vectors: Sequence[Sequence[float]]) -> list[int]:
        if self.cluster_method == "kmeans":
            k = self.n_clusters or max(1, int(math.sqrt(max(1, len(vectors)))))
            try:
                return _kmeans_cluster(vectors, k=k, seed=self.seed)
            except ImportError as exc:
                log.warning("falling back to greedy clustering: %s", exc)
        return _greedy_cluster(vectors, threshold=self.similarity_threshold)

    def _build_seeds(
        self,
        items: Sequence[dict[str, Any]],
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> list[Seed]:
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(lab, []).append(idx)

        seeds: list[Seed] = []
        # Order clusters by descending size so the resulting seeds.jsonl puts the
        # most common intents first — easier to inspect by hand.
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for new_id, (_, indices) in enumerate(ordered):
            rep_idx = _pick_representative(texts, vectors, indices)
            rep_item = items[rep_idx]
            domain = rep_item["domain"]
            lineage_ids = [items[i]["lineage_id"] for i in indices if items[i].get("lineage_id")]
            session_ids = [items[i]["session_id"] for i in indices if items[i].get("session_id")]
            seed_id = self._stable_seed_id(rep_item["query"], new_id)
            seeds.append(
                Seed(
                    seed_id=seed_id,
                    query=rep_item["query"],
                    cluster_id=new_id,
                    cluster_size=len(indices),
                    original_lineage_ids=lineage_ids,
                    domain=domain,
                    source_session_ids=session_ids,
                )
            )
        return seeds

    @staticmethod
    def _stable_seed_id(query: str, cluster_id: int) -> str:
        h = hashlib.sha256(f"{cluster_id}::{query}".encode()).hexdigest()
        return f"seed-{h[:12]}"

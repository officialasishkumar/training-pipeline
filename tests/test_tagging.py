"""Trajectory tagging and stratified split."""

from __future__ import annotations

import pytest

from training_pipeline.schemas.events import (
    Trajectory,
)
from training_pipeline.tagging.complexity import (
    classify_complexity,
    compute_tags,
    tag_trajectory,
)
from training_pipeline.tagging.stratify import stratified_split, stratum_key


def test_qa_trajectory_is_easy_or_below(qa_trajectory: Trajectory):
    tags = compute_tags(qa_trajectory)
    assert tags.complexity_band in ("trivial", "easy")
    assert tags.has_recovery is False


def test_agent_trajectory_recovers(agent_trajectory: Trajectory):
    tags = compute_tags(agent_trajectory)
    assert tags.has_recovery is True
    assert tags.num_tool_calls == 2
    assert tags.distinct_tools == 1
    assert tags.complexity_band in ("medium", "hard", "extreme")


def test_classify_band_boundaries():
    assert classify_complexity(0.0) == "trivial"
    assert classify_complexity(2.0) == "easy"
    assert classify_complexity(3.5) == "medium"
    assert classify_complexity(6.0) == "hard"
    assert classify_complexity(8.0) == "extreme"


def test_tag_trajectory_attaches_complexity(agent_trajectory: Trajectory):
    out = tag_trajectory(agent_trajectory)
    assert "complexity" in out.tags
    assert "complexity_band" in out.tags["complexity"]


def test_stratum_key_uses_complexity_and_domain(qa_trajectory: Trajectory):
    tagged = tag_trajectory(qa_trajectory)
    key = stratum_key(tagged)
    assert "agronomy" in key


def test_stratified_split_balances_strata(qa_trajectory: Trajectory, agent_trajectory: Trajectory):
    qa_tagged = tag_trajectory(qa_trajectory)
    ag_tagged = tag_trajectory(agent_trajectory)

    # 20 trajectories: 10 easy, 10 hard.
    items = []
    for i in range(10):
        items.append(qa_tagged.model_copy(update={"session_id": f"q-{i}"}))
        items.append(ag_tagged.model_copy(update={"session_id": f"a-{i}"}))

    s = stratified_split(items, fractions=(0.7, 0.15, 0.15), seed=1)
    assert len(s.train) + len(s.val) + len(s.test) == len(items)
    # Both strata should have at least one item in train.
    train_strata = {stratum_key(items[i]) for i in s.train}
    assert len(train_strata) == 2


def test_stratified_split_is_deterministic(qa_trajectory: Trajectory):
    tagged = tag_trajectory(qa_trajectory)
    items = [tagged.model_copy(update={"session_id": f"q-{i}"}) for i in range(20)]
    s1 = stratified_split(items, seed=42)
    s2 = stratified_split(items, seed=42)
    assert s1.train == s2.train
    assert s1.val == s2.val


def test_stratified_split_validates_fractions():
    with pytest.raises(ValueError):
        stratified_split([], fractions=(0.5, 0.4, 0.0), seed=0)  # sums to 0.9

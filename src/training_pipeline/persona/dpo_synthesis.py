"""Synthesise DPO preference pairs from three complementary sources.

DPO needs ``(prompt, chosen, rejected)`` triples. Production logs alone
give us at most a thin slice of explicit feedback pairs, and the mentor's
target is ~20k pairs against a quality > scale bar. We close the gap
with two synthesis modes that *generate* the rejected side from a known
chosen side:

* **real_pairs** — for a seed cluster with both successful and failed
  trajectories, pair them. No model calls, just bucketing.
* **persona_violation** — take a chosen trajectory and prompt the
  generator with "rewrite this response to violate <rule>". The
  rewrite is the rejected side. The pair is stamped with the rule id
  so a downstream trainer can up-weight pairs that target rules the
  scorer flags as currently weak.
* **tool_inefficiency** — take an n-tool successful trajectory and
  generate an (n+k) variant that arrives at the same answer the long
  way. The pair is stamped with the inefficiency type
  (``extra_tool_calls`` / ``redundant_call`` / ``wrong_tool_first``).

Each synthesised pair carries enough metadata in
``pair_metadata`` to drive curriculum-style sampling at training time.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from training_pipeline.export.sft import trajectory_to_messages
from training_pipeline.persona.loader import Persona, Rule
from training_pipeline.persona.scorer import (
    PersonaScorer,
    TrajectoryScore,
    _final_assistant_text,
)
from training_pipeline.schemas.events import (
    ToolCallEvent,
    Trajectory,
)
from training_pipeline.schemas.exports import DPORecord, SFTMessage, SFTToolCall

log = logging.getLogger(__name__)


class PreferencePairSource(str, enum.Enum):
    """Where a preference pair came from."""

    REAL_PAIRS = "real_pairs"
    PERSONA_VIOLATION = "persona_violation"
    TOOL_INEFFICIENCY = "tool_inefficiency"


class InefficiencyType(str, enum.Enum):
    EXTRA_TOOL_CALLS = "extra_tool_calls"
    REDUNDANT_CALL = "redundant_call"
    WRONG_TOOL_FIRST = "wrong_tool_first"


# ---------------------------------------------------------------------------
# Rewrite providers — pluggable so callers can swap in a real LLM.
# ---------------------------------------------------------------------------


class RewriteProvider(Protocol):
    """Synthesises a *rejected* response given a chosen one and a directive."""

    def rewrite_to_violate(
        self,
        *,
        chosen_text: str,
        rule: Rule,
        prompt_messages: list[dict[str, Any]],
    ) -> str: ...

    def rewrite_inefficient(
        self,
        *,
        chosen_text: str,
        inefficiency: InefficiencyType,
        prompt_messages: list[dict[str, Any]],
    ) -> str: ...


@dataclass
class StubRewriteProvider:
    """Deterministic, dependency-free rewrite generator.

    Real runs should pass an LLM-backed provider. The stub is good enough
    for tests, smoke runs, and pipeline plumbing — it injects controlled
    perturbations that the persona scorer can be made to flag.
    """

    def rewrite_to_violate(
        self,
        *,
        chosen_text: str,
        rule: Rule,
        prompt_messages: list[dict[str, Any]],
    ) -> str:
        # Deliberately strip likely persona signals so we have a clear contrast.
        # The exact transformations are chosen to fail common rule patterns:
        #   * remove citations / links so 'must hyperlink' rules fire
        #   * remove the first sentence so 'must include greeting' fires
        #   * append a forbidden tone marker so 'no medical advice' style
        #     rules can fire
        text = chosen_text
        text = _strip_links(text)
        text = _strip_first_sentence(text)
        return f"{text} (Note: this is the only definitive answer.)"

    def rewrite_inefficient(
        self,
        *,
        chosen_text: str,
        inefficiency: InefficiencyType,
        prompt_messages: list[dict[str, Any]],
    ) -> str:
        # Tool-inefficiency pairs are pre-built upstream from the trajectory's
        # extra-step variant; the rewrite provider only needs to produce a
        # surface text. We keep the same answer content so the chosen/rejected
        # contrast is purely on the path, not the outcome.
        return chosen_text


def _strip_links(text: str) -> str:
    import re

    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)


def _strip_first_sentence(text: str) -> str:
    parts = text.split(". ", 1)
    return parts[1] if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------


@dataclass
class PreferencePairBuilder:
    """Build (chosen, rejected) DPO pairs from scored trajectories."""

    persona: Persona | None = None
    scorer: PersonaScorer | None = None
    rewrite_provider: RewriteProvider = field(default_factory=StubRewriteProvider)
    system_prompt: str | None = None
    judged_pass_threshold: float = 0.6
    """A trajectory with persona score below this is treated as a *failed*
    candidate when looking for real_pairs partners."""

    def build(
        self,
        trajectories: Iterable[Trajectory],
        *,
        sources: Iterable[PreferencePairSource] | None = None,
    ) -> Iterator[DPORecord]:
        srcs = list(sources or list(PreferencePairSource))
        items = self._materialise(list(trajectories))

        if PreferencePairSource.REAL_PAIRS in srcs:
            yield from self._real_pairs(items)
        if PreferencePairSource.PERSONA_VIOLATION in srcs:
            yield from self._persona_violation(items)
        if PreferencePairSource.TOOL_INEFFICIENCY in srcs:
            yield from self._tool_inefficiency(items)

    # -- materialise --------------------------------------------------------

    def _materialise(
        self, trajectories: list[Trajectory]
    ) -> list[tuple[Trajectory, TrajectoryScore | None]]:
        """Score each trajectory once if a scorer is available."""
        if self.scorer is None:
            return [(t, None) for t in trajectories]
        return [(t, self.scorer.score(t)) for t in trajectories]

    # -- real_pairs ---------------------------------------------------------

    def _real_pairs(
        self, items: list[tuple[Trajectory, TrajectoryScore | None]]
    ) -> Iterator[DPORecord]:
        """Pair successful trajectories against failed ones for the same seed cluster.

        Success / failure can come from two signals:
          * persona score (when a scorer is configured): score >= threshold = success.
          * trajectory tags: ``synthetic.finish_reason == 'final'`` and no errors.
        """
        buckets: dict[str, dict[str, list[Trajectory]]] = {}
        for traj, score in items:
            bucket_key = _seed_cluster_key(traj)
            outcome = self._classify_outcome(traj, score)
            buckets.setdefault(bucket_key, {"success": [], "failure": []})[outcome].append(traj)

        for cluster_key, sides in buckets.items():
            wins = sides["success"]
            losses = sides["failure"]
            if not wins or not losses:
                continue
            # One pair per (best_win, worst_loss) — keep the pair count modest
            # so a 1000-trajectory cluster doesn't fan out to 1M pairs.
            for chosen_traj in wins[:1]:
                for rejected_traj in losses[:1]:
                    yield self._dpo_from_two_trajectories(
                        chosen_traj=chosen_traj,
                        rejected_traj=rejected_traj,
                        source=PreferencePairSource.REAL_PAIRS,
                        extra_meta={"cluster": cluster_key},
                    )

    def _classify_outcome(
        self, traj: Trajectory, score: TrajectoryScore | None
    ) -> str:
        if score is not None:
            return "success" if score.score >= self.judged_pass_threshold else "failure"
        synth = traj.tags.get("synthetic") if isinstance(traj.tags, dict) else None
        if isinstance(synth, dict):
            if synth.get("finish_reason") == "final":
                return "success"
            return "failure"
        # Fall back to error count: any tool error => failure side.
        if any(getattr(e, "is_error", False) for e in traj.events):
            return "failure"
        return "success"

    # -- persona_violation --------------------------------------------------

    def _persona_violation(
        self, items: list[tuple[Trajectory, TrajectoryScore | None]]
    ) -> Iterator[DPORecord]:
        if self.persona is None or not self.persona.rules:
            return
        for traj, score in items:
            chosen_text = _final_assistant_text(traj)
            if not chosen_text:
                continue
            # Skip already-failing trajectories — pair generation needs a
            # solid chosen side.
            if score is not None and not score.hard_pass:
                continue
            base_msgs = trajectory_to_messages(traj, system_prompt=self.system_prompt)
            prompt_msgs, last_assistant_msgs = _split_prompt_and_last_assistant(base_msgs)
            chosen_msg = (
                last_assistant_msgs[-1]
                if last_assistant_msgs
                else SFTMessage(role="assistant", content=chosen_text)
            )
            for rule in self.persona.rules:
                rejected_text = self.rewrite_provider.rewrite_to_violate(
                    chosen_text=chosen_text,
                    rule=rule,
                    prompt_messages=[m.model_dump(exclude_none=True) for m in prompt_msgs],
                )
                if not rejected_text or rejected_text.strip() == chosen_text.strip():
                    continue
                meta: dict[str, Any] = {
                    "source": PreferencePairSource.PERSONA_VIOLATION.value,
                    "session_id": traj.session_id,
                    "violation_rule_id": rule.id,
                    "violation_rule_severity": rule.severity.value,
                }
                if traj.lineage_id is not None:
                    meta["lineage_id"] = traj.lineage_id
                yield DPORecord(
                    prompt=prompt_msgs,
                    chosen=[chosen_msg],
                    rejected=[SFTMessage(role="assistant", content=rejected_text)],
                    metadata=meta,
                )

    # -- tool_inefficiency --------------------------------------------------

    def _tool_inefficiency(
        self, items: list[tuple[Trajectory, TrajectoryScore | None]]
    ) -> Iterator[DPORecord]:
        for traj, _ in items:
            n_calls = sum(1 for e in traj.events if isinstance(e, ToolCallEvent))
            if n_calls < 1:
                continue
            chosen_text = _final_assistant_text(traj)
            if not chosen_text:
                continue
            base_msgs = trajectory_to_messages(traj, system_prompt=self.system_prompt)
            prompt_msgs, last_assistant_msgs = _split_prompt_and_last_assistant(base_msgs)
            chosen_msg = (
                last_assistant_msgs[-1]
                if last_assistant_msgs
                else SFTMessage(role="assistant", content=chosen_text)
            )
            inefficiency = (
                InefficiencyType.EXTRA_TOOL_CALLS
                if n_calls <= 2
                else InefficiencyType.REDUNDANT_CALL
            )
            distinct = {c.name for e in traj.events if isinstance(e, ToolCallEvent) for c in e.tool_calls}
            if len(distinct) >= 2:
                inefficiency = InefficiencyType.WRONG_TOOL_FIRST

            rejected_text = self.rewrite_provider.rewrite_inefficient(
                chosen_text=chosen_text,
                inefficiency=inefficiency,
                prompt_messages=[m.model_dump(exclude_none=True) for m in prompt_msgs],
            )
            if not rejected_text:
                continue

            # The rejected side is the *same* answer reached via more steps.
            # We approximate that by replaying the prompt with synthetic extra
            # tool messages prepended to the rejected assistant turn — enough
            # signal for DPO to prefer the shorter path.
            extra_tool_msgs = _synthesise_extra_tool_messages(traj, count=2)
            rejected_msgs: list[SFTMessage] = [
                *extra_tool_msgs,
                SFTMessage(role="assistant", content=rejected_text),
            ]

            meta = {
                "source": PreferencePairSource.TOOL_INEFFICIENCY.value,
                "session_id": traj.session_id,
                "inefficiency_type": inefficiency.value,
                "n_tool_calls_chosen": n_calls,
                "n_tool_calls_rejected": n_calls + len(extra_tool_msgs) // 2,
            }
            if traj.lineage_id is not None:
                meta["lineage_id"] = traj.lineage_id
            yield DPORecord(
                prompt=prompt_msgs,
                chosen=[chosen_msg],
                rejected=rejected_msgs,
                metadata=meta,
            )

    # -- helpers ------------------------------------------------------------

    def _dpo_from_two_trajectories(
        self,
        *,
        chosen_traj: Trajectory,
        rejected_traj: Trajectory,
        source: PreferencePairSource,
        extra_meta: dict[str, Any] | None = None,
    ) -> DPORecord:
        chosen_msgs = trajectory_to_messages(chosen_traj, system_prompt=self.system_prompt)
        rejected_msgs = trajectory_to_messages(rejected_traj, system_prompt=self.system_prompt)
        prompt_msgs, chosen_tail = _split_prompt_and_last_assistant(chosen_msgs)
        _, rejected_tail = _split_prompt_and_last_assistant(rejected_msgs)
        if not chosen_tail:
            chosen_tail = [
                SFTMessage(role="assistant", content=_final_assistant_text(chosen_traj))
            ]
        if not rejected_tail:
            rejected_tail = [
                SFTMessage(
                    role="assistant",
                    content=_final_assistant_text(rejected_traj)
                    or "I'm sorry, I'm unable to complete the request.",
                )
            ]
        meta: dict[str, Any] = {
            "source": source.value,
            "chosen_session_id": chosen_traj.session_id,
            "rejected_session_id": rejected_traj.session_id,
            **(extra_meta or {}),
        }
        if chosen_traj.lineage_id is not None:
            meta["chosen_lineage_id"] = chosen_traj.lineage_id
        if rejected_traj.lineage_id is not None:
            meta["rejected_lineage_id"] = rejected_traj.lineage_id
        return DPORecord(
            prompt=prompt_msgs,
            chosen=chosen_tail,
            rejected=rejected_tail,
            metadata=meta,
        )


def _seed_cluster_key(traj: Trajectory) -> str:
    """Stable cluster key for grouping trajectories.

    Synthetic trajectories carry the seed lineage; real trajectories don't,
    so we fall back to the trajectory fingerprint over the user turn.
    """
    if traj.lineage_id:
        return f"lineage:{traj.lineage_id}"
    synth = traj.tags.get("synthetic") if isinstance(traj.tags, dict) else None
    if isinstance(synth, dict) and synth.get("seed_id"):
        return f"seed:{synth['seed_id']}"
    return f"fp:{traj.fingerprint()}"


def _split_prompt_and_last_assistant(
    messages: list[SFTMessage],
) -> tuple[list[SFTMessage], list[SFTMessage]]:
    """Walk back to the last assistant turn so the prompt ends right before it."""
    last_assistant_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        return list(messages), []
    return list(messages[:last_assistant_idx]), [messages[last_assistant_idx]]


def _synthesise_extra_tool_messages(
    traj: Trajectory, *, count: int
) -> list[SFTMessage]:
    """Build a small chain of synthetic tool call/result messages.

    Used to give the *rejected* side of a tool-inefficiency pair a longer
    path. We don't need plausible answers — just plausible structure that
    DPO can learn to disprefer.
    """
    msgs: list[SFTMessage] = []
    base_call_id = f"redund_{traj.session_id[:6]}"
    tool_name = "lookup"
    for ev in traj.events:
        if isinstance(ev, ToolCallEvent) and ev.tool_calls:
            tool_name = ev.tool_calls[0].name
            break
    for i in range(count):
        cid = f"{base_call_id}_{i}"
        msgs.append(
            SFTMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    SFTToolCall(
                        id=cid,
                        name=tool_name,
                        arguments={"redundant_check": True, "step": i},
                    )
                ],
            )
        )
        msgs.append(
            SFTMessage(
                role="tool",
                tool_call_id=cid,
                name=tool_name,
                content='{"redundant": true}',
            )
        )
    return msgs


# ---------------------------------------------------------------------------
# Convenience for the existing DPO export — produce records for a single
# trajectory regardless of source so the export layer can iterate without
# knowing about persona / efficiency strategies.
# ---------------------------------------------------------------------------


def synthesise_pairs(
    trajectories: Iterable[Trajectory],
    *,
    persona: Persona | None,
    scorer: PersonaScorer | None,
    sources: Iterable[PreferencePairSource] | None = None,
    rewrite_provider: RewriteProvider | None = None,
    system_prompt: str | None = None,
) -> Iterator[DPORecord]:
    builder = PreferencePairBuilder(
        persona=persona,
        scorer=scorer,
        rewrite_provider=rewrite_provider or StubRewriteProvider(),
        system_prompt=system_prompt,
    )
    yield from builder.build(trajectories, sources=sources)

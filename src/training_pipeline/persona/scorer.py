"""Run programmatic + LLM-judge persona rules against a trajectory.

The scorer is intentionally conservative on what it considers
"assistant text" — we strip tool-call envelopes and only consider the
final natural-language replies. That matches what a human reviewer
would judge ("does this *answer* respect the persona?") rather than
penalising tool wiring details.

LLM-judge rules call out to an injected ``LLMJudge`` implementation;
the default ``StubJudge`` is a deterministic, offline stub that
returns "pass" by convention so the rest of the pipeline is testable
without a model. Production runs should pass a real judge.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from training_pipeline.persona.loader import (
    LLMJudgeRule,
    Persona,
    ProgrammaticRule,
    RuleSeverity,
)
from training_pipeline.schemas.events import (
    AssistantEvent,
    Trajectory,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    rule_text: str
    severity: str
    passed: bool
    score: float
    """Continuous 0..1 score. Hard fails are always 0.0."""
    reasons: list[str] = field(default_factory=list)
    evaluator: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "severity": self.severity,
            "passed": self.passed,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "evaluator": self.evaluator,
        }


@dataclass
class TrajectoryScore:
    score: float
    """Weighted aggregate in 0..1; hard-rule failures dominate."""
    hard_pass: bool
    """False iff at least one hard rule failed."""
    rule_results: list[RuleResult]
    reasons: list[str]
    """Flat de-duplicated list of failure reasons. Empty iff every rule passes."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "hard_pass": self.hard_pass,
            "rule_results": [r.as_dict() for r in self.rule_results],
            "reasons": list(self.reasons),
        }


class LLMJudge(Protocol):
    """Minimal protocol for an LLM-judge implementation.

    Production-ready implementations should accept the rule text + the
    candidate assistant text and return a ``(score, reason)`` tuple. For
    this prototype we keep it simple — the protocol matches what the stub
    backend already provides.
    """

    def evaluate(
        self,
        *,
        rule_id: str,
        criterion: str,
        assistant_text: str,
        full_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[float, str]:
        """Return ``(score in 0..1, reason)``."""
        ...


@dataclass
class StubJudge:
    """Deterministic offline judge.

    Default verdict is **pass** — we don't want CI runs to flap based on
    cosmetic text differences. Override the ``always_pass`` flag, the
    ``forbid_substrings`` set, or pass a custom ``decide`` callable when a
    test needs a controlled fail.
    """

    always_pass: bool = True
    forbid_substrings: tuple[str, ...] = ()
    """If any of these substrings appears in the assistant text the judge
    fails the rule. Cheap way to simulate a judge without a real model."""

    def evaluate(
        self,
        *,
        rule_id: str,
        criterion: str,
        assistant_text: str,
        full_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[float, str]:
        if not self.always_pass:
            return (0.0, f"stub judge configured to fail rule on criterion: {criterion!r}")
        text_lc = assistant_text.lower()
        for sub in self.forbid_substrings:
            if sub.lower() in text_lc:
                return (0.0, f"forbidden substring {sub!r} present in assistant text")
        return (1.0, "stub judge: pass")


@dataclass
class TransformersJudge:  # pragma: no cover — exercised under the [generate] extra
    """Real LLM judge backed by ``transformers``. Lazy-loaded.

    The judge is given the rule criterion and the assistant text, and is
    asked to answer ``YES``/``NO`` plus a one-line reason. We parse the
    first line for the verdict.
    """

    model_id: str
    device: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 64

    _model: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. "
                "Install with `pip install training-pipeline[generate]`."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.device:
            self._model = self._model.to(self.device)

    def evaluate(
        self,
        *,
        rule_id: str,
        criterion: str,
        assistant_text: str,
        full_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[float, str]:
        self._load()
        prompt = (
            "You are evaluating an assistant response against a persona rule.\n\n"
            f"RULE [{rule_id}]: {criterion}\n\n"
            f"ASSISTANT RESPONSE: {assistant_text}\n\n"
            "Reply with one line: PASS or FAIL, then a brief reason."
        )
        chat = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(chat, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-3),
        )
        completion = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        first_line = completion.splitlines()[0] if completion else ""
        verdict = first_line.upper().startswith("PASS")
        return (1.0 if verdict else 0.0, first_line or "no response")


@dataclass
class PersonaScorer:
    """Score trajectories against a persona."""

    persona: Persona
    judge: LLMJudge = field(default_factory=StubJudge)
    hard_weight: float = 1.0
    """Hard-rule failures zero out the aggregate score."""
    soft_weight: float = 1.0
    """Each soft rule contributes equally."""

    def score(self, trajectory: Trajectory) -> TrajectoryScore:
        assistant_text = _final_assistant_text(trajectory)
        results: list[RuleResult] = []
        for rule in self.persona.rules:
            if isinstance(rule, ProgrammaticRule):
                results.append(self._score_programmatic(rule, assistant_text))
            elif isinstance(rule, LLMJudgeRule):
                results.append(self._score_judge(rule, assistant_text, trajectory))
            else:  # pragma: no cover
                log.warning("unsupported rule type: %s", type(rule).__name__)

        return self._aggregate(results)

    def score_many(
        self, trajectories: Iterable[Trajectory]
    ) -> Iterable[tuple[Trajectory, TrajectoryScore]]:
        for traj in trajectories:
            yield traj, self.score(traj)

    def annotate(self, trajectory: Trajectory) -> Trajectory:
        """Return a copy with the persona score attached under ``tags['persona']``."""
        s = self.score(trajectory)
        new_tags = {**trajectory.tags, "persona": s.as_dict()}
        return trajectory.model_copy(update={"tags": new_tags})

    def _score_programmatic(
        self, rule: ProgrammaticRule, assistant_text: str
    ) -> RuleResult:
        passed, reasons = rule.evaluate(assistant_text)
        return RuleResult(
            rule_id=rule.id,
            rule_text=rule.text,
            severity=rule.severity.value,
            passed=passed,
            score=1.0 if passed else 0.0,
            reasons=reasons,
            evaluator="programmatic",
        )

    def _score_judge(
        self,
        rule: LLMJudgeRule,
        assistant_text: str,
        trajectory: Trajectory,
    ) -> RuleResult:
        msgs = _trajectory_messages_for_judge(trajectory)
        try:
            score, reason = self.judge.evaluate(
                rule_id=rule.id,
                criterion=rule.criterion or rule.text,
                assistant_text=assistant_text,
                full_messages=msgs,
            )
        except Exception as exc:  # pragma: no cover — judge contract
            log.warning("judge raised on rule %s: %s", rule.id, exc)
            score, reason = (0.5, f"judge error: {exc!r}")
        passed = score >= 0.5
        return RuleResult(
            rule_id=rule.id,
            rule_text=rule.text,
            severity=rule.severity.value,
            passed=passed,
            score=float(score),
            reasons=[] if passed else [reason],
            evaluator="judge",
        )

    def _aggregate(self, results: list[RuleResult]) -> TrajectoryScore:
        if not results:
            return TrajectoryScore(score=1.0, hard_pass=True, rule_results=[], reasons=[])
        hard_failures = [r for r in results if r.severity == RuleSeverity.HARD.value and not r.passed]
        soft_results = [r for r in results if r.severity == RuleSeverity.SOFT.value]
        soft_score = (
            sum(r.score for r in soft_results) / len(soft_results)
            if soft_results
            else 1.0
        )
        score = 0.0 if hard_failures else soft_score
        reasons = [reason for r in results for reason in r.reasons]
        return TrajectoryScore(
            score=score,
            hard_pass=not hard_failures,
            rule_results=results,
            reasons=reasons,
        )


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def _final_assistant_text(trajectory: Trajectory) -> str:
    """Concatenate assistant *text* turns (skipping tool-call envelopes)."""
    parts: list[str] = []
    for ev in trajectory.events:
        if isinstance(ev, AssistantEvent) and ev.content:
            cleaned = _TOOL_CALL_TAG_RE.sub("", ev.content).strip()
            if cleaned:
                parts.append(cleaned)
    return "\n".join(parts)


def _trajectory_messages_for_judge(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Render a trajectory as a chat-message list the judge can read."""
    out: list[dict[str, Any]] = []
    for ev in trajectory.events:
        if isinstance(ev, AssistantEvent) and ev.content:
            out.append({"role": "assistant", "content": ev.content})
        elif ev.__class__.__name__ == "UserEvent":
            out.append({"role": "user", "content": getattr(ev, "content", "")})
    return out

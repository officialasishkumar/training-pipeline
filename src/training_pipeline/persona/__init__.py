"""Persona-grounded scoring and DPO pair synthesis.

The mentor's framing on the OpenAgriNet RFP is that the bot's persona
matters more than aggregate accuracy: vernacular speech, hyperlinked
references, no-medical-advice rules, and so on. This package makes the
persona checkable.

* ``loader.parse_persona`` parses a markdown persona file into rules
  partitioned by evaluator type — programmatic (regex / structural)
  and LLM-judge (free-text criteria).
* ``scorer.PersonaScorer`` runs every rule against a trajectory and
  returns a structured verdict (per-rule pass/fail + aggregate score).
* ``dpo_synthesis.PreferencePairBuilder`` produces (chosen, rejected)
  pairs from three sources: real successful-vs-failed trajectories,
  deliberately persona-violating rewrites, and tool-inefficiency
  rewrites that arrive at the same answer the long way.

Programmatic rules are pure-Python checks. LLM-judge rules call a
configurable judge model — default is the same Qwen2.5-7B used for
generation, with the option to swap in a stronger judge.
"""

from training_pipeline.persona.dpo_synthesis import (
    InefficiencyType,
    PreferencePairBuilder,
    PreferencePairSource,
)
from training_pipeline.persona.loader import (
    LLMJudgeRule,
    Persona,
    ProgrammaticRule,
    Rule,
    RuleSeverity,
    parse_persona,
)
from training_pipeline.persona.scorer import (
    LLMJudge,
    PersonaScorer,
    RuleResult,
    StubJudge,
    TrajectoryScore,
)

__all__ = [
    "InefficiencyType",
    "LLMJudge",
    "LLMJudgeRule",
    "Persona",
    "PersonaScorer",
    "PreferencePairBuilder",
    "PreferencePairSource",
    "ProgrammaticRule",
    "Rule",
    "RuleResult",
    "RuleSeverity",
    "StubJudge",
    "TrajectoryScore",
    "parse_persona",
]

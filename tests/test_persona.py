"""Persona loader, scorer, and DPO pair synthesizer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from training_pipeline.cli import app
from training_pipeline.persona.dpo_synthesis import (
    InefficiencyType,
    PreferencePairBuilder,
    PreferencePairSource,
    StubRewriteProvider,
)
from training_pipeline.persona.loader import (
    LLMJudgeRule,
    ProgrammaticRule,
    RuleSeverity,
    parse_persona,
    persona_to_dict,
)
from training_pipeline.persona.scorer import (
    PersonaScorer,
    StubJudge,
)
from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_parse_persona_minimal():
    md = """# A Persona

## Voice
- Speak concisely.
- [hard][regex: \\[[^\\]]+\\]\\(https?://[^)]+\\)] References must be hyperlinks.
- [forbid: \\b(synergy|leverage)\\b] No corporate jargon.
- [contains: ₹] Numerical answers must include the rupee symbol.
"""
    p = parse_persona(md, name="test")
    assert p.name == "test"
    assert len(p.rules) == 4
    by_evaluator = {type(r).__name__ for r in p.rules}
    assert by_evaluator == {"ProgrammaticRule", "LLMJudgeRule"}

    hard = [r for r in p.rules if r.severity is RuleSeverity.HARD]
    assert len(hard) == 1
    assert isinstance(hard[0], ProgrammaticRule)


def test_parse_persona_section_path():
    md = "# Top\n\n## Voice\n\n### Tone\n- Speak warmly.\n"
    p = parse_persona(md)
    assert any("voice" in r.section for r in p.rules)


def test_parse_persona_explicit_id_resolves():
    md = "## A\n- [id: must-cite] Must cite.\n- [id: must-cite] Must cite again.\n"
    p = parse_persona(md)
    assert {r.id for r in p.rules} == {"must-cite", "must-cite-2"}


def test_parse_persona_judge_default():
    p = parse_persona("## V\n- Be empathetic.\n")
    assert isinstance(p.rules[0], LLMJudgeRule)


def test_parse_persona_programmatic_can_override_to_judge():
    p = parse_persona("## V\n- [regex: foo][judge] Match foo.\n")
    assert isinstance(p.rules[0], LLMJudgeRule)


def test_parse_persona_from_file(tmp_path: Path):
    f = tmp_path / "p.md"
    f.write_text("## V\n- Always cite sources.\n")
    p = parse_persona(f)
    assert p.name == "p"
    assert len(p.rules) == 1


def test_persona_to_dict_summarises_counts():
    md = "## V\n- [hard] R1.\n- [judge] R2.\n- [regex: \\d] R3.\n"
    p = parse_persona(md)
    summary = persona_to_dict(p)
    assert summary["n_rules"] == 3
    assert summary["n_hard"] == 1
    assert summary["n_judge"] >= 1
    assert summary["n_programmatic"] >= 1


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def _ts(s: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, s, tzinfo=timezone.utc)


def _traj(text: str, sid: str = "s") -> Trajectory:
    return Trajectory(
        session_id=sid,
        domain="agronomy",
        events=[
            UserEvent(event_id="u", session_id=sid, timestamp=_ts(0), content="hi"),
            AssistantEvent(
                event_id="a", session_id=sid, timestamp=_ts(1), content=text
            ),
        ],
    )


def test_scorer_passes_when_programmatic_satisfied():
    md = (
        "## V\n"
        "- [regex: \\bcite\\b] Must mention cite.\n"
        "- [forbid: \\bjargon\\b] No jargon.\n"
    )
    p = parse_persona(md)
    scorer = PersonaScorer(persona=p, judge=StubJudge())
    s = scorer.score(_traj("Please cite this source."))
    assert s.score == 1.0
    assert s.hard_pass


def test_scorer_fails_hard_rule_zeros_score():
    md = "## V\n- [hard][regex: \\bcite\\b] Must cite.\n- [contains: hello] Greet.\n"
    p = parse_persona(md)
    scorer = PersonaScorer(persona=p, judge=StubJudge())
    s = scorer.score(_traj("hello, no source here"))
    assert not s.hard_pass
    assert s.score == 0.0
    assert any("missing required pattern" in r for r in s.reasons)


def test_scorer_soft_rule_failure_lowers_score():
    md = "## V\n- [contains: ₹] Include rupee.\n- [contains: kg/ha] Include unit.\n"
    p = parse_persona(md)
    scorer = PersonaScorer(persona=p, judge=StubJudge())
    s = scorer.score(_traj("Apply 50 kg/ha urea."))
    # 1 of 2 soft rules pass.
    assert 0.0 < s.score < 1.0
    assert s.hard_pass


def test_scorer_uses_judge_for_freeform_rule():
    md = "## V\n- Speak warmly.\n"
    p = parse_persona(md)
    scorer = PersonaScorer(
        persona=p, judge=StubJudge(forbid_substrings=("cold",))
    )
    s_pass = scorer.score(_traj("Welcome, friend!"))
    assert s_pass.score == 1.0
    s_fail = scorer.score(_traj("Cold response."))
    assert s_fail.score < 1.0


def test_scorer_skips_tool_call_envelopes():
    """Tool-call markup shouldn't satisfy or violate text rules."""
    md = "## V\n- [forbid: tool_call] No tool envelopes.\n"
    p = parse_persona(md)
    traj = _traj("This is plain text. <tool_call>{...}</tool_call>")
    scorer = PersonaScorer(persona=p, judge=StubJudge())
    s = scorer.score(traj)
    assert s.hard_pass  # the envelope is stripped before scoring


def test_scorer_annotate_attaches_tags():
    md = "## V\n- Be helpful.\n"
    p = parse_persona(md)
    scorer = PersonaScorer(persona=p, judge=StubJudge())
    annotated = scorer.annotate(_traj("OK"))
    assert "persona" in annotated.tags
    assert "score" in annotated.tags["persona"]


# ---------------------------------------------------------------------------
# DPO synthesis
# ---------------------------------------------------------------------------


def _agent_traj(*, sid: str, n_calls: int, success: bool) -> Trajectory:
    events: list = [
        UserEvent(event_id="u", session_id=sid, timestamp=_ts(0), content="market price"),
    ]
    base_t = 1
    for i in range(n_calls):
        events.append(
            ToolCallEvent(
                event_id=f"tc{i}",
                session_id=sid,
                timestamp=_ts(base_t),
                tool_calls=[ToolCall(id=f"c{i}", name="mandi_price", arguments={"x": i})],
            )
        )
        base_t += 1
        events.append(
            ToolResultEvent(
                event_id=f"tr{i}",
                session_id=sid,
                timestamp=_ts(base_t),
                tool_call_id=f"c{i}",
                name="mandi_price",
                content="ok" if success else "error",
                is_error=not success,
            )
        )
        base_t += 1
    events.append(
        AssistantEvent(
            event_id="af",
            session_id=sid,
            timestamp=_ts(base_t),
            content="The mandi price is around 1500 per quintal. [USDA](https://example.com)",
        )
    )
    return Trajectory(
        session_id=sid,
        domain="agronomy",
        lineage_id=f"seed-{sid}",
        events=events,
        tags={"synthetic": {"finish_reason": "final" if success else "max_steps"}},
    )


def test_real_pairs_emits_when_both_outcomes_present():
    win = _agent_traj(sid="A", n_calls=1, success=True)
    loss = _agent_traj(sid="B", n_calls=1, success=False)
    # Both share lineage so they cluster.
    loss = loss.model_copy(update={"lineage_id": win.lineage_id})
    builder = PreferencePairBuilder()
    pairs = list(builder.build([win, loss], sources=[PreferencePairSource.REAL_PAIRS]))
    assert len(pairs) == 1
    assert pairs[0].metadata["source"] == PreferencePairSource.REAL_PAIRS.value
    assert pairs[0].metadata["chosen_session_id"] == "A"


def test_real_pairs_skips_when_only_one_outcome():
    a = _agent_traj(sid="A", n_calls=1, success=True)
    b = _agent_traj(sid="B", n_calls=1, success=True)
    builder = PreferencePairBuilder()
    pairs = list(builder.build([a, b], sources=[PreferencePairSource.REAL_PAIRS]))
    assert pairs == []


def test_persona_violation_stamps_rule_id():
    md = "## V\n- [hard][regex: foo] R1.\n- R2 free form.\n"
    persona = parse_persona(md)
    builder = PreferencePairBuilder(persona=persona, rewrite_provider=StubRewriteProvider())
    win = _agent_traj(sid="A", n_calls=1, success=True)
    pairs = list(builder.build([win], sources=[PreferencePairSource.PERSONA_VIOLATION]))
    rule_ids = {p.metadata["violation_rule_id"] for p in pairs}
    assert rule_ids == {r.id for r in persona.rules}
    for p in pairs:
        assert p.metadata["source"] == PreferencePairSource.PERSONA_VIOLATION.value


def test_persona_violation_skips_without_persona():
    builder = PreferencePairBuilder(persona=None)
    win = _agent_traj(sid="A", n_calls=1, success=True)
    pairs = list(builder.build([win], sources=[PreferencePairSource.PERSONA_VIOLATION]))
    assert pairs == []


def test_tool_inefficiency_stamps_inefficiency_type():
    win = _agent_traj(sid="A", n_calls=1, success=True)
    builder = PreferencePairBuilder()
    pairs = list(builder.build([win], sources=[PreferencePairSource.TOOL_INEFFICIENCY]))
    assert len(pairs) == 1
    p = pairs[0]
    assert p.metadata["source"] == PreferencePairSource.TOOL_INEFFICIENCY.value
    assert p.metadata["inefficiency_type"] in {x.value for x in InefficiencyType}
    assert p.metadata["n_tool_calls_chosen"] == 1
    # Rejected has more steps.
    assert len(p.rejected) > len(p.chosen)


def test_tool_inefficiency_skips_pure_qa():
    qa = Trajectory(
        session_id="qa",
        events=[
            UserEvent(event_id="u", session_id="qa", timestamp=_ts(0), content="hi"),
            AssistantEvent(event_id="a", session_id="qa", timestamp=_ts(1), content="hello"),
        ],
    )
    builder = PreferencePairBuilder()
    pairs = list(builder.build([qa], sources=[PreferencePairSource.TOOL_INEFFICIENCY]))
    assert pairs == []


def test_all_sources_combine():
    md = "## V\n- Be helpful.\n"
    persona = parse_persona(md)
    win = _agent_traj(sid="A", n_calls=2, success=True)
    loss = _agent_traj(sid="B", n_calls=1, success=False)
    loss = loss.model_copy(update={"lineage_id": win.lineage_id})
    builder = PreferencePairBuilder(persona=persona)
    pairs = list(builder.build([win, loss]))
    sources = {p.metadata["source"] for p in pairs}
    assert PreferencePairSource.REAL_PAIRS.value in sources
    assert PreferencePairSource.PERSONA_VIOLATION.value in sources
    assert PreferencePairSource.TOOL_INEFFICIENCY.value in sources


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_score_writes_persona_tags(tmp_path: Path, runner: CliRunner):
    persona = tmp_path / "persona.md"
    persona.write_text("## V\n- [contains: cite] Must mention cite.\n")
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(_traj("Please cite the source.", sid="s1").model_dump(mode="json", exclude_none=True))
        + "\n"
    )
    out = tmp_path / "scored.jsonl"
    result = runner.invoke(
        app,
        [
            "score",
            "--persona",
            str(persona),
            "--input",
            str(src),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = [
        json.loads(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["tags"]["persona"]["score"] == 1.0


def test_cli_dpo_synthesize_real_pairs(tmp_path: Path, runner: CliRunner):
    win = _agent_traj(sid="A", n_calls=1, success=True)
    loss = _agent_traj(sid="B", n_calls=1, success=False).model_copy(
        update={"lineage_id": win.lineage_id}
    )
    src = tmp_path / "in.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(t.model_dump(mode="json", exclude_none=True)) for t in [win, loss]
        )
        + "\n"
    )
    out = tmp_path / "pairs.jsonl"
    result = runner.invoke(
        app,
        [
            "dpo",
            "synthesize",
            "--input",
            str(src),
            "--output",
            str(out),
            "--strategy",
            "real_pairs",
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = [
        json.loads(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["metadata"]["source"] == "real_pairs"


def test_cli_export_dpo_legacy_strategy_still_works(tmp_path: Path, runner: CliRunner):
    """The new strategies don't break the legacy --strategy feedback path."""
    traj = _traj("ok", sid="s")
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(traj.model_dump(mode="json", exclude_none=True)) + "\n")
    out_dir = tmp_path / "dpo"
    result = runner.invoke(
        app,
        [
            "export",
            "dpo",
            "--input",
            str(src),
            "--output-dir",
            str(out_dir),
            "--strategy",
            "feedback",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # No feedback_pairs in the trajectory ⇒ no records, but the command should still succeed.
    assert (out_dir / "dataset_card.json").exists()


def test_cli_export_dpo_persona_strategy_routes_to_synthesizer(
    tmp_path: Path, runner: CliRunner
):
    persona = tmp_path / "persona.md"
    persona.write_text("## V\n- Helpful tone.\n")
    win = _agent_traj(sid="A", n_calls=1, success=True)
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(win.model_dump(mode="json", exclude_none=True)) + "\n")
    out_dir = tmp_path / "dpo"
    result = runner.invoke(
        app,
        [
            "export",
            "dpo",
            "--input",
            str(src),
            "--output-dir",
            str(out_dir),
            "--strategy",
            "persona_violation",
            "--persona",
            str(persona),
        ],
    )
    assert result.exit_code == 0, result.stdout
    shards = list(out_dir.glob("dpo-*.jsonl"))
    assert shards


def test_example_persona_parses():
    """The shipped example persona must always parse cleanly."""
    persona = parse_persona(Path("examples/persona.example.md"))
    assert persona.rules
    # Hard rules exist so the rubric has something to fail on.
    assert any(r.severity is RuleSeverity.HARD for r in persona.rules)
    # At least one programmatic rule and one judge rule.
    assert any(isinstance(r, ProgrammaticRule) for r in persona.rules)
    assert any(isinstance(r, LLMJudgeRule) for r in persona.rules)

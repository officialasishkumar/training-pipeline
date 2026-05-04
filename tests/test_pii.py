"""PII detection, redaction, and audit sampling."""

from __future__ import annotations

from pathlib import Path

from training_pipeline.pii.audit import AuditSampler
from training_pipeline.pii.redactor import Redactor, redact_trajectory
from training_pipeline.pii.rules import (
    detect_all,
    load_rules,
)
from training_pipeline.schemas.events import (
    ToolCall,
    ToolCallEvent,
    Trajectory,
    UserEvent,
)


def test_email_detection():
    text = "Reach me at user.name@example.org."
    dets = detect_all(text)
    assert any(d.category == "EMAIL" for d in dets)


def test_credit_card_luhn_filters_invalid():
    # Random non-Luhn-valid 16-digit number.
    text = "card 1234567890123456"
    dets = detect_all(text)
    assert not any(d.category == "CREDIT_CARD" for d in dets)


def test_credit_card_luhn_keeps_valid():
    # Test PAN that passes Luhn (Visa test card).
    text = "4111-1111-1111-1111"
    dets = detect_all(text)
    assert any(d.category == "CREDIT_CARD" for d in dets)


def test_indian_phone():
    text = "Call me at 9876543210 or +91 98765 43210."
    dets = detect_all(text)
    phones = [d for d in dets if d.category == "PHONE"]
    assert len(phones) >= 1


def test_aadhaar_detection():
    text = "ID: 1234 5678 9012"
    dets = detect_all(text)
    assert any(d.category == "GOV_ID_IN" for d in dets)


def test_pan_detection():
    text = "PAN ABCDE1234F belongs to me."
    dets = detect_all(text)
    assert any(d.rule == "pan" for d in dets)


def test_redactor_consistent_placeholders():
    r = Redactor()
    out, _ = r.redact_text("Email me at a@b.com")
    assert "a@b.com" not in out
    # Run inside a single shared memo to check stability.
    memo: dict[str, str] = {}
    counts: dict[str, int] = {}
    o1, _ = r.redact_text("a@b.com", memo=memo, category_counts=counts)
    o2, _ = r.redact_text("a@b.com appears again", memo=memo, category_counts=counts)
    assert o1 == "[EMAIL_1]"
    assert "[EMAIL_1]" in o2  # same placeholder


def test_redactor_different_emails_get_different_placeholders():
    r = Redactor()
    memo: dict[str, str] = {}
    counts: dict[str, int] = {}
    o1, _ = r.redact_text("a@b.com", memo=memo, category_counts=counts)
    o2, _ = r.redact_text("c@d.com", memo=memo, category_counts=counts)
    assert o1 == "[EMAIL_1]"
    assert o2 == "[EMAIL_2]"


def test_redact_trajectory_marks_pii_redacted_tag(agent_trajectory: Trajectory):
    res = redact_trajectory(agent_trajectory)
    assert res.trajectory.tags["pii_redacted"] is True
    # The user message had an email — should be redacted.
    user = next(e for e in res.trajectory.events if isinstance(e, UserEvent))
    assert "@example.com" not in user.content


def test_redactor_preserves_tool_call_arg_structure():
    """Tool args must remain valid JSON dicts even after redaction."""
    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(event_id="u", session_id="s", content="hi"),
            ToolCallEvent(
                event_id="tc",
                session_id="s",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="email_send",
                        arguments={"to": "user@example.com", "subject": "ok"},
                    )
                ],
            ),
        ],
    )
    res = redact_trajectory(traj)
    tc = res.trajectory.events[1]
    assert isinstance(tc, ToolCallEvent)
    args = tc.tool_calls[0].arguments
    assert isinstance(args, dict)
    assert "@example.com" not in args.get("to", "")


def test_audit_sampler_is_deterministic():
    """Same seed + same records → same sample."""
    a = AuditSampler(rate=0.5, seed=42, cap=100)
    b = AuditSampler(rate=0.5, seed=42, cap=100)
    records = [{"session_id": str(i), "x": i} for i in range(200)]
    for r in records:
        a.consider(r)
        b.consider(r)
    assert [r["x"] for r in a.consume()] == [r["x"] for r in b.consume()]


def test_load_rules_extends_builtins(tmp_path: Path):
    p = tmp_path / "rules.yaml"
    p.write_text(
        "rules:\n  - name: emp_id\n    category: INTERNAL_ID\n    pattern: 'EMP-\\d{6}'\n",
        encoding="utf-8",
    )
    rules = load_rules(p)
    assert any(r.name == "emp_id" for r in rules)
    assert any(r.name == "email" for r in rules)


def test_load_rules_disable_builtins(tmp_path: Path):
    p = tmp_path / "rules.yaml"
    p.write_text(
        "include_builtins: false\n"
        "rules:\n"
        "  - name: emp_id\n"
        "    category: INTERNAL_ID\n"
        "    pattern: 'EMP-\\d{6}'\n",
        encoding="utf-8",
    )
    rules = load_rules(p)
    assert {r.name for r in rules} == {"emp_id"}


def test_verify_redacted_no_false_positive_on_placeholders(agent_trajectory: Trajectory):
    """The leakage check must not classify [EMAIL_1] as a leak of itself."""
    res = redact_trajectory(agent_trajectory)
    assert res.has_leaks is False
    assert "pii_leak_count" not in res.trajectory.tags


def test_verify_redacted_catches_rule_with_hole():
    """A redactor whose rule has a hole should flag the leftover as a leak.

    Build a deliberately narrow rule (only matches a literal "secret-1") and
    redact a payload containing "secret-1" *and* "secret-2". After redaction
    "secret-2" survives. The same narrow rule, re-run, *still* finds it — so
    the verifier must surface the leak.
    """
    from training_pipeline.pii.rules import PIIRule

    narrow = PIIRule(
        name="literal_one",
        category="SECRET",
        pattern=r"secret-1",
        placeholder="[SECRET]",
        flags=0,
    )
    fuller = PIIRule(
        name="any_secret",
        category="SECRET",
        pattern=r"secret-\d",
        placeholder="[SECRET]",
        flags=0,
    )
    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(
                event_id="u",
                session_id="s",
                content="here is secret-1 and also secret-2",
            ),
        ],
    )
    # Redact with the narrow rule — secret-2 survives.
    redacted = Redactor(rules=(narrow,)).redact_trajectory(traj).trajectory
    assert "secret-2" in redacted.events[0].content
    # Verify with the fuller rule — leak surfaces.
    leaks = Redactor(rules=(fuller,)).verify_redacted(list(redacted.events))
    assert any(lk.category == "SECRET" and "secret-2" in lk.text for lk in leaks)


def test_verify_redacted_handles_tool_call_args(agent_trajectory: Trajectory):
    """Leakage check should re-scan tool-call argument JSON, not just text."""

    traj = Trajectory(
        session_id="s",
        events=[
            UserEvent(event_id="u", session_id="s", content="hi"),
            ToolCallEvent(
                event_id="tc",
                session_id="s",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="email_send",
                        arguments={"to": "user@example.com", "subject": "ok"},
                    )
                ],
            ),
        ],
    )
    # Use a redactor with no rules — nothing gets redacted.
    no_rules = Redactor(rules=())
    redacted = no_rules.redact_trajectory(traj, verify=False).trajectory
    # Verify with the full rule set — should see the surviving email in args.
    full = Redactor()
    leaks = full.verify_redacted(list(redacted.events))
    assert any(lk.category == "EMAIL" and "tool_calls[0].arguments" in lk.field for lk in leaks)


def test_lineage_id_set_by_canonical_adapter():
    """Trajectories from canonical records get a content-derived lineage id."""
    from training_pipeline.ingest.sources import from_canonical

    record = {
        "session_id": "s",
        "events": [
            {"kind": "user", "event_id": "u1", "session_id": "s", "content": "hi"}
        ],
    }
    traj = from_canonical(record)
    assert traj.lineage_id and traj.lineage_id.startswith("lin_")
    assert all(ev.lineage_id == traj.lineage_id for ev in traj.events)
    # Re-running the adapter on the same record yields the same id.
    again = from_canonical(record)
    assert again.lineage_id == traj.lineage_id


def test_lineage_propagates_through_redaction():
    from training_pipeline.ingest.sources import from_canonical

    record = {
        "session_id": "s",
        "events": [
            {
                "kind": "user",
                "event_id": "u1",
                "session_id": "s",
                "content": "Email me at user@example.com",
            }
        ],
    }
    traj = from_canonical(record)
    res = redact_trajectory(traj)
    assert res.trajectory.lineage_id == traj.lineage_id
    assert all(ev.lineage_id == traj.lineage_id for ev in res.trajectory.events)

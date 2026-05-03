"""PII detection, redaction, and audit sampling."""

from __future__ import annotations

from pathlib import Path

import pytest

from training_pipeline.pii.audit import AuditSampler
from training_pipeline.pii.redactor import Redactor, redact_trajectory
from training_pipeline.pii.rules import (
    BUILTIN_RULES,
    detect_all,
    load_rules,
)
from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
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
    out2, _ = r.redact_text("Email a@b.com again")
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
        "rules:\n"
        "  - name: emp_id\n"
        "    category: INTERNAL_ID\n"
        "    pattern: 'EMP-\\d{6}'\n",
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

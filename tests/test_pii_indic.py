"""Multilingual PII orchestration tests.

All inputs are synthetic — placeholder names ("Suresh", "Lakshmi"),
fake-but-syntactically-valid Aadhaar numbers (Verhoeff-checksummed),
and made-up Indian mobile numbers. No real PII is ever stored in the
test fixtures.
"""

from __future__ import annotations

import pytest

from training_pipeline.pii.orchestrator import (
    FieldRuleEngine,
    IndianIDEngine,
    PIIOrchestrator,
    RegexRuleEngine,
    coverage_report,
    detect_language,
)
from training_pipeline.pii.recognizers.indian_ids import (
    aadhaar_check_digit,
    is_valid_aadhaar,
    is_valid_pan,
)
from training_pipeline.pii.structured_fields import (
    detect_structured_fields,
    label_groups,
)

# ---------------------------------------------------------------------------
# Helpers — synthesise valid Aadhaar / PAN for testing.
# ---------------------------------------------------------------------------


def _make_aadhaar(prefix: str = "23456789012") -> str:
    """Return a 12-digit string with a valid Verhoeff check digit."""
    cd = aadhaar_check_digit(prefix)
    return f"{prefix}{cd}"


@pytest.fixture
def valid_aadhaar() -> str:
    return _make_aadhaar("23456789012")


@pytest.fixture
def valid_pan() -> str:
    # 4th char must be in PFCATBLJGH; "P" = individual.
    return "ABCPE1234F"


# ---------------------------------------------------------------------------
# Verhoeff Aadhaar
# ---------------------------------------------------------------------------


def test_verhoeff_accepts_valid_aadhaar(valid_aadhaar: str):
    assert is_valid_aadhaar(valid_aadhaar)


def test_verhoeff_rejects_one_digit_off(valid_aadhaar: str):
    bad_digit = "1" if valid_aadhaar[-1] != "1" else "2"
    bad = valid_aadhaar[:-1] + bad_digit
    if bad == valid_aadhaar:
        return  # extremely unlikely; skip
    assert not is_valid_aadhaar(bad)


def test_verhoeff_rejects_leading_zero():
    """Aadhaar can't start with 0 or 1 by spec."""
    cd = aadhaar_check_digit("01234567890")
    candidate = f"01234567890{cd}"
    assert not is_valid_aadhaar(candidate)


def test_verhoeff_rejects_wrong_length():
    assert not is_valid_aadhaar("12345")
    assert not is_valid_aadhaar("23456789012345")


def test_aadhaar_with_separators_validates(valid_aadhaar: str):
    spaced = f"{valid_aadhaar[:4]} {valid_aadhaar[4:8]} {valid_aadhaar[8:]}"
    dashed = f"{valid_aadhaar[:4]}-{valid_aadhaar[4:8]}-{valid_aadhaar[8:]}"
    assert is_valid_aadhaar(spaced)
    assert is_valid_aadhaar(dashed)


# ---------------------------------------------------------------------------
# PAN
# ---------------------------------------------------------------------------


def test_pan_individual_accepted():
    assert is_valid_pan("ABCPE1234F")  # 4th char P


def test_pan_unknown_entity_class_rejected():
    # 4th char Z is not in the recognised set.
    assert not is_valid_pan("ABCZE1234F")


def test_pan_wrong_shape_rejected():
    assert not is_valid_pan("ABC1234567")
    # Validator uppercases before checking shape, so lowercase input with the
    # right shape and a known entity-class char is still accepted.
    assert is_valid_pan("abcpe1234f")


# ---------------------------------------------------------------------------
# Indian ID engine
# ---------------------------------------------------------------------------


def test_indian_id_engine_finds_aadhaar(valid_aadhaar: str):
    text = f"Farmer profile shows Aadhaar {valid_aadhaar} and crop area 2 ha."
    hits = IndianIDEngine().detect(text)
    aadhaar_hits = [h for h in hits if h.rule == "aadhaar_verhoeff"]
    assert aadhaar_hits
    assert aadhaar_hits[0].text.replace(" ", "").replace("-", "") == valid_aadhaar


def test_indian_id_engine_skips_invalid_aadhaar():
    """A 12-digit number with a wrong checksum must NOT be detected as Aadhaar."""
    bad = "234567890123"
    assert not is_valid_aadhaar(bad)
    text = f"reference id {bad}"
    hits = IndianIDEngine().detect(text)
    assert not any(h.rule == "aadhaar_verhoeff" for h in hits)


def test_indian_id_engine_finds_pan(valid_pan: str):
    text = f"PAN: {valid_pan} on file"
    hits = IndianIDEngine().detect(text)
    assert any(h.rule == "pan_with_checksum" and h.text == valid_pan for h in hits)


def test_indian_id_engine_finds_indian_mobile():
    text = "Call me at +91 9876543210 anytime"
    hits = IndianIDEngine().detect(text)
    assert any(h.rule == "indian_mobile" for h in hits)


def test_indian_id_engine_does_not_misclassify_landline():
    """Landlines (start with 0-5) should not match the mobile rule."""
    text = "Office phone is 0 22 1234 5678"
    hits = IndianIDEngine().detect(text)
    assert not any(h.rule == "indian_mobile" for h in hits)


# ---------------------------------------------------------------------------
# Structured field rules
# ---------------------------------------------------------------------------


def test_field_rules_english_labels():
    text = "Name: Suresh Patil\nMobile: 9876543210\nAadhaar: 1234 5678 9012"
    hits = detect_structured_fields(text)
    rules = {h.rule for h in hits}
    assert {"field_name", "field_phone", "field_aadhaar"} <= rules


def test_field_rules_label_does_not_overlap_value():
    """Detection must cover the value, never the label itself."""
    text = "Name: Suresh"
    [hit] = [h for h in detect_structured_fields(text) if h.rule == "field_name"]
    assert hit.text == "Suresh"
    assert text[hit.start : hit.end] == "Suresh"


def test_field_rules_hindi_labels():
    text = "नाम: सुरेश पाटिल\nमोबाइल: 9876543210\nआधार: 234567890123"
    hits = detect_structured_fields(text)
    rules = {h.rule for h in hits}
    assert "field_name" in rules
    assert "field_phone" in rules
    assert "field_aadhaar" in rules


def test_field_rules_tamil_labels():
    text = "பெயர்: லட்சுமி\nஆதார்: 1234-5678-9012"
    hits = detect_structured_fields(text)
    rules = {h.rule for h in hits}
    assert "field_name" in rules
    assert "field_aadhaar" in rules


def test_field_rules_marathi_address():
    text = "गाँव: पुणे, पता: फ्लॅट 12, मेन रोड"
    hits = detect_structured_fields(text)
    addr_hits = [h for h in hits if h.rule == "field_address"]
    assert addr_hits
    assert addr_hits[0].text


def test_field_rules_max_value_chars_bounded():
    """A missing newline shouldn't cause unbounded gobbling."""
    text = "Name: " + "x" * 500
    hits = [h for h in detect_structured_fields(text) if h.rule == "field_name"]
    assert hits
    assert len(hits[0].text) <= 80


def test_field_rules_label_groups_listed():
    groups = list(label_groups())
    assert "field_name" in groups
    assert "field_phone" in groups


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_default_pipeline_finds_indian_pii(valid_aadhaar: str):
    text = (
        f"Farmer Suresh, Aadhaar {valid_aadhaar}, mobile +91 9876543210. "
        "Email: officer@example.com"
    )
    orch = PIIOrchestrator.default(enable_presidio=False, enable_indicner=False)
    hits = orch.detect(text)
    categories = {h.category for h in hits}
    assert {"GOV_ID_IN", "PHONE", "EMAIL"} <= categories


def test_orchestrator_dedupes_overlapping_engines():
    """Indian-ID and field-rule engines both target Aadhaar; orchestrator
    must collapse the overlap to a single decision."""
    aadhaar = _make_aadhaar("87654321098")
    text = f"Aadhaar: {aadhaar}"
    orch = PIIOrchestrator(engines=[IndianIDEngine(), FieldRuleEngine()])
    hits = orch.detect(text)
    aadhaar_hits = [h for h in hits if aadhaar in h.text or h.category == "GOV_ID_IN"]
    # Single span — earlier engine (IndianIDEngine) wins.
    starts = {h.start for h in aadhaar_hits}
    assert len(starts) == 1


def test_orchestrator_engine_findings_carry_engine_name(valid_aadhaar: str):
    text = f"Aadhaar: {valid_aadhaar}"
    orch = PIIOrchestrator(engines=[IndianIDEngine(), FieldRuleEngine()])
    findings = orch.detect_with_engines(text)
    assert findings
    assert findings[0].engine == "indian_ids"


def test_orchestrator_regex_engine_skips_govid():
    """RegexRuleEngine drops GOV_ID_IN by default so it doesn't shadow
    the validated Indian-ID engine."""
    text = "PAN: ABCPE1234F"
    hits = RegexRuleEngine().detect(text)
    assert not any(h.category == "GOV_ID_IN" for h in hits)


def test_detect_language_devanagari_is_hindi():
    assert detect_language("नाम सुरेश") == "hi"


def test_detect_language_tamil():
    assert detect_language("பெயர் லட்சுமி") == "ta"


def test_detect_language_english_default():
    assert detect_language("Hello world") == "en"
    assert detect_language("") == "en"


def test_coverage_report_compares_engines(valid_aadhaar: str):
    """The presidio-only path is skipped when presidio isn't installed,
    but the orchestrator path still finds the Aadhaar."""
    samples = [
        ("aadhaar_in_english", f"Aadhaar: {valid_aadhaar}"),
        ("hindi_name", "नाम: सुरेश"),
    ]
    expected = {
        "aadhaar_in_english": {"GOV_ID_IN"},
        "hindi_name": {"PERSON"},
    }
    rep = coverage_report(samples, expected_categories=expected)
    rows = {r["sample"]: r for r in rep["rows"]}
    # Full orchestrator should hit Aadhaar via IndianIDEngine.
    assert "GOV_ID_IN" in rows["aadhaar_in_english"]["full_orchestrator"]
    # Hindi name caught via field rules at minimum.
    assert "PERSON" in rows["hindi_name"]["full_orchestrator"]


# ---------------------------------------------------------------------------
# Integration with redactor — orchestrator detections feed back through
# the existing rules pipeline indirectly via the PII rule mechanism.
# ---------------------------------------------------------------------------


def test_orchestrator_does_not_double_count(valid_aadhaar: str, valid_pan: str):
    text = f"Aadhaar: {valid_aadhaar}, PAN: {valid_pan}"
    orch = PIIOrchestrator.default(enable_presidio=False, enable_indicner=False)
    hits = orch.detect(text)
    aadhaar_spans = [h for h in hits if h.text.replace(" ", "").replace("-", "") == valid_aadhaar]
    pan_spans = [h for h in hits if h.text == valid_pan]
    assert len(aadhaar_spans) == 1
    assert len(pan_spans) == 1


def test_orchestrator_handles_mixed_language_text(valid_aadhaar: str):
    text = (
        "नाम: सुरेश पाटिल, "
        f"Mobile: +91 9876543210, Aadhaar: {valid_aadhaar}, "
        "Email: officer@example.com"
    )
    orch = PIIOrchestrator.default(enable_presidio=False, enable_indicner=False)
    hits = orch.detect(text)
    categories = {h.category for h in hits}
    assert "GOV_ID_IN" in categories
    assert "PHONE" in categories
    assert "EMAIL" in categories
    # Field-rule fallback catches the Hindi name even without IndicNER.
    person_or_field = [h for h in hits if h.rule == "field_name"]
    assert person_or_field


def test_back_compat_ner_import():
    """The legacy import path still works."""
    from training_pipeline.pii.ner import PresidioDetector  # noqa: F401

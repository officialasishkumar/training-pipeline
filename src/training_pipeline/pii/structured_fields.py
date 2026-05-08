"""Field-rule fallback for tool-return content.

Production tool returns from agristack-style services are usually
structured: ``"Name: <value>"``, ``"Mobile: <value>"``, ``"Aadhaar:
<value>"``. IndicNER can miss the *name value* (because the surrounding
context isn't a sentence) and Presidio can miss the *Mobile* and
*Aadhaar* labels because they look like field labels rather than NER
triggers.

This module pattern-matches the labels themselves — language-agnostic
key tokens like ``Name:``/``नाम:``/``பெயர்:`` followed by the value.
The detected span covers the value only, never the label, so
``[NAME]: [PHONE]`` is the natural redacted form.

The intent is *defense-in-depth*: if Presidio + IndicNER catches a
field, great. If they miss it, this module catches it via the label.
The orchestrator deduplicates overlapping detections.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import regex as re

from training_pipeline.pii.rules import PIIDetection

# ---------------------------------------------------------------------------
# Field labels in English + Indic scripts.
# ---------------------------------------------------------------------------


_NAME_LABELS: tuple[str, ...] = (
    r"name",
    r"full\s+name",
    r"farmer\s+name",
    r"applicant",
    r"नाम",          # Hindi/Marathi
    r"पूरा\s+नाम",   # Hindi
    r"નામ",          # Gujarati
    r"নাম",          # Bengali
    r"ਨਾਮ",          # Punjabi
    r"ಹೆಸರು",        # Kannada
    r"പേര്",         # Malayalam
    r"పేరు",         # Telugu
    r"பெயர்",        # Tamil
)

_PHONE_LABELS: tuple[str, ...] = (
    r"mobile",
    r"phone",
    r"contact(?:\s+(?:no\.?|number))?",
    r"मोबाइल",       # Hindi
    r"फ़ोन",         # Hindi (variant)
    r"फोन",          # Hindi
    r"মোবাইল",       # Bengali
    r"মোবাইল\s+ন",   # Bengali variant
    r"ਫੋਨ",          # Punjabi
    r"ஃபோன்",        # Tamil
    r"ఫోన్",          # Telugu
    r"ಫೋನ್",        # Kannada
)

_AADHAAR_LABELS: tuple[str, ...] = (
    r"aadhaar",
    r"aadhar",
    r"uid",
    r"आधार",         # Hindi/Marathi
    r"ਆਧਾਰ",         # Punjabi
    r"আধার",         # Bengali
    r"ஆதார்",        # Tamil
    r"ఆధార్",         # Telugu
    r"ಆಧಾರ್",       # Kannada
)

_DOB_LABELS: tuple[str, ...] = (
    r"dob",
    r"date\s+of\s+birth",
    r"birth\s+date",
    r"जन्म\s+तिथि",  # Hindi
    r"जन्म\s+ता",    # Marathi short
    r"ജനന\s+തീയതി", # Malayalam
    r"பிறந்த\s+தேதி",  # Tamil
)

_PAN_LABELS: tuple[str, ...] = (r"pan", r"पैन", r"ਪੈਨ", r"প্যান", r"పాన్")
_VOTER_LABELS: tuple[str, ...] = (r"voter\s*id", r"epic", r"वोटर")
_DL_LABELS: tuple[str, ...] = (
    r"driving\s+licen[cs]e",
    r"dl\s+no",
    r"ड्राइविंग",   # Hindi
)
_ADDRESS_LABELS: tuple[str, ...] = (
    r"address",
    r"village",
    r"पता",          # Hindi/Marathi
    r"गाँव",         # Hindi village
    r"গ্রাম",         # Bengali village
    r"ગામ",          # Gujarati village
    r"ஊர்",          # Tamil village/town
)


@dataclass(frozen=True)
class _LabelGroup:
    name: str
    labels: tuple[str, ...]
    category: str
    placeholder: str
    # Maximum number of characters of value text we'll redact. Stops a
    # missed line-break from dragging in a whole paragraph.
    max_value_chars: int = 80


_GROUPS: tuple[_LabelGroup, ...] = (
    _LabelGroup("field_name", _NAME_LABELS, "PERSON", "[NAME]"),
    _LabelGroup("field_phone", _PHONE_LABELS, "PHONE", "[PHONE]"),
    _LabelGroup("field_aadhaar", _AADHAAR_LABELS, "GOV_ID_IN", "[AADHAAR]", 32),
    _LabelGroup("field_pan", _PAN_LABELS, "GOV_ID_IN", "[PAN]", 16),
    _LabelGroup("field_voter", _VOTER_LABELS, "GOV_ID_IN", "[VOTER_ID]", 24),
    _LabelGroup("field_dl", _DL_LABELS, "GOV_ID_IN", "[DRIVING_LICENSE]", 32),
    _LabelGroup("field_dob", _DOB_LABELS, "DATE_TIME", "[DOB]", 24),
    _LabelGroup("field_address", _ADDRESS_LABELS, "LOCATION", "[ADDRESS]"),
)


def _build_pattern(group: _LabelGroup) -> re.Pattern[str]:
    """``Label[: =-]\\s*VALUE`` where VALUE stops at end-of-line or "," / ";".

    The ``max_value_chars`` is enforced via a bounded ``{0,N}`` quantifier
    so a missed line break doesn't cause the detector to gobble.
    """
    label_alt = "|".join(group.labels)
    bound = group.max_value_chars
    return re.compile(
        rf"(?P<label>(?:{label_alt}))\s*[:\-=]\s*(?P<value>[^\r\n,;]{{1,{bound}}})",
        re.IGNORECASE | re.UNICODE,
    )


_COMPILED: tuple[tuple[_LabelGroup, re.Pattern[str]], ...] = tuple(
    (g, _build_pattern(g)) for g in _GROUPS
)


def detect_structured_fields(text: str) -> list[PIIDetection]:
    """Return one ``PIIDetection`` per ``Label: value`` match in ``text``.

    The detection's span covers the *value*, not the label. Languages
    are not provided as a separate parameter — patterns are
    multilingual by construction.
    """
    out: list[PIIDetection] = []
    for group, pattern in _COMPILED:
        for m in pattern.finditer(text):
            value = m.group("value").strip()
            if not value:
                continue
            value_start = m.start("value") + (len(m.group("value")) - len(m.group("value").lstrip()))
            value_end = value_start + len(value)
            out.append(
                PIIDetection(
                    rule=group.name,
                    category=group.category,
                    start=value_start,
                    end=value_end,
                    text=value,
                )
            )
    out.sort(key=lambda d: (d.start, -d.end))
    return out


def label_groups() -> Iterable[str]:
    """Names of the field-label groups (for diagnostic listing)."""
    return (g.name for g in _GROUPS)

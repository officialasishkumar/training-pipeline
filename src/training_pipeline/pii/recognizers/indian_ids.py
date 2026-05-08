"""Indian-government-ID recognizers with proper checksum validation.

These recognizers extend the regex-only ``BUILTIN_RULES`` with proper
validation so we stop classifying random 12-digit numbers as Aadhaar.
The mentor's PII coverage matrix specifically called out phone numbers
and Indian govt IDs — these are the highest-risk categories in
agricultural traces because farmer profiles routinely contain them.

* **Aadhaar** — 12 digits, Verhoeff checksum mandatory. UIDAI's spec
  is ``Verhoeff(d1..d11, d12)`` where d12 is the check digit. The
  built-in regex-only rule is replaced by one that validates.
* **PAN** — ``5L 4D 1L`` shape. The 4th letter encodes entity type
  (P=individual, F=firm, etc.); we validate it's one of the known
  set, which kills the false-positive rate compared to a pure regex.
* **Indian mobile** — 10 digits starting 6-9, with optional ``+91`` /
  ``0`` prefix. Bound check stops it from gobbling part of an Aadhaar.
* **Voter ID (EPIC)** — 3 letters + 7 digits.
* **Driving Licence** — 2-letter state code + (`-`/space optional) 13
  digits or 2-letter state code + ``RR`` + ``YYYY`` + 7-digit serial.

The Verhoeff implementation is the table-based form so both validation
*and* check-digit generation are available for synthetic test fixtures.
"""

from __future__ import annotations

import regex as re

from training_pipeline.pii.rules import PIIRule

# ---------------------------------------------------------------------------
# Verhoeff checksum (used for Aadhaar)
# ---------------------------------------------------------------------------

# Multiplication table for the dihedral group D5 (used by Verhoeff).
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
# Permutation table.
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
# Inverse of the D table — used by the check-digit generator.
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _verhoeff_checksum(digits: list[int]) -> int:
    """Return 0 when ``digits`` (already including the check digit) is valid."""
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c


def is_valid_aadhaar(value: str) -> bool:
    """Return True if ``value`` is a 12-digit string with valid Verhoeff."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 12:
        return False
    # Aadhaar can't start with 0 or 1.
    if digits[0] in (0, 1):
        return False
    return _verhoeff_checksum(digits) == 0


def aadhaar_check_digit(first_eleven: str) -> int:
    """Compute the Verhoeff check digit for the first 11 digits.

    Useful for generating synthetic-but-syntactically-valid test fixtures
    in tests/test_pii_indic.py.
    """
    digits = [int(c) for c in first_eleven if c.isdigit()]
    if len(digits) != 11:
        raise ValueError("aadhaar_check_digit needs exactly 11 digits")
    # Standard generator: append a 0 placeholder, run the same loop, invert.
    c = 0
    for i, d in enumerate(reversed([*digits, 0])):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return _VERHOEFF_INV[c]


# ---------------------------------------------------------------------------
# PAN
# ---------------------------------------------------------------------------

# Fourth character encodes entity class:
#   P=Individual, F=Firm, C=Company, A=Association of Persons,
#   T=Trust, B=Body of Individuals, L=Local Authority,
#   J=Artificial Juridical Person, G=Government, H=HUF.
_PAN_ENTITY_TYPES = frozenset("PFCATBLJGH")


def is_valid_pan(value: str) -> bool:
    """Validate the 4th-character entity-type rule that bare regex misses."""
    cleaned = value.upper().strip()
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", cleaned):
        return False
    return cleaned[3] in _PAN_ENTITY_TYPES


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

AADHAAR_RULE = PIIRule(
    name="aadhaar_verhoeff",
    category="GOV_ID_IN",
    pattern=r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)",
    placeholder="[AADHAAR]",
    description="12-digit Aadhaar number; checksum validated outside the regex.",
)

PAN_RULE = PIIRule(
    name="pan_with_checksum",
    category="GOV_ID_IN",
    pattern=r"\b[A-Z]{5}\d{4}[A-Z]\b",
    placeholder="[PAN]",
    description="PAN with 4th-character entity-class validation.",
)

INDIAN_MOBILE_RULE = PIIRule(
    name="indian_mobile",
    category="PHONE",
    pattern=r"(?<!\d)(?:\+91[\s\-]?|0)?[6-9]\d{9}(?!\d)",
    placeholder="[PHONE]",
    description="Indian mobile (+91/0 prefix; 10 digits starting 6-9).",
)

VOTER_ID_RULE = PIIRule(
    name="voter_id_epic",
    category="GOV_ID_IN",
    pattern=r"\b[A-Z]{3}\d{7}\b",
    placeholder="[VOTER_ID]",
    description="EPIC (Election Commission) voter ID.",
)

DRIVING_LICENSE_RULE = PIIRule(
    name="driving_license_in",
    category="GOV_ID_IN",
    # State (2L) + RTO (2D) + (year 4D) + 7-digit serial, common forms:
    #   "MH-12 19840012345" / "MH1219840012345" / "DL01 20240001234"
    pattern=r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?(?:19|20)\d{2}[\s\-]?\d{7}\b",
    placeholder="[DRIVING_LICENSE]",
    description="Indian driving licence (state + RTO + year + serial).",
)


INDIAN_ID_RULES: tuple[PIIRule, ...] = (
    AADHAAR_RULE,
    PAN_RULE,
    INDIAN_MOBILE_RULE,
    VOTER_ID_RULE,
    DRIVING_LICENSE_RULE,
)

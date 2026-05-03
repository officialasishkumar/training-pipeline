"""Rule-based PII detection.

Each ``PIIRule`` is a named regex with a category and a placeholder template.
The default rule set covers the categories we expect in OpenAgriNet (and most
agentic-LLM logs):

- email, phone (E.164 + IN), credit card, IBAN
- IPv4 / IPv6, URLs with credentials
- GPS coordinates and lat/lng decimals (agriculture context)
- India-specific: Aadhaar, PAN, GSTIN
- generic: sequences that look like API keys or AWS credentials

This module is *deterministic* — no ML, no surprises, easy to audit. For
PERSON / ORGANIZATION / LOCATION-type entities, plug in
``training_pipeline.pii.ner.PresidioDetector`` (optional dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex as re
import yaml


@dataclass(frozen=True)
class PIIRule:
    name: str
    category: str
    pattern: str
    placeholder: str = "[REDACTED:{category}]"
    flags: int = re.IGNORECASE
    luhn_check: bool = False  # for credit card validation
    description: str = ""

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True)
class PIIDetection:
    rule: str
    category: str
    start: int
    end: int
    text: str


# ----- Luhn check (for credit cards) ------------------------------------------------


def _luhn_valid(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ----- Built-in rules ---------------------------------------------------------------

BUILTIN_RULES: tuple[PIIRule, ...] = (
    PIIRule(
        name="email",
        category="EMAIL",
        pattern=r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
        placeholder="[EMAIL]",
    ),
    PIIRule(
        name="phone_e164",
        category="PHONE",
        # E.164 (+ digits, optional spaces/dashes) — bounded to avoid catching part of long IDs.
        pattern=r"(?<!\d)\+\d{1,3}[\s\-.]?(?:\(?\d{1,4}\)?[\s\-.]?){1,4}\d{2,4}(?!\d)",
        placeholder="[PHONE]",
    ),
    PIIRule(
        name="phone_in",
        category="PHONE",
        # India 10-digit mobile (starts with 6-9), with optional 0/91 prefix.
        pattern=r"(?<!\d)(?:0|91)?[\s\-]?[6-9]\d{9}(?!\d)",
        placeholder="[PHONE]",
    ),
    PIIRule(
        name="credit_card",
        category="CREDIT_CARD",
        pattern=r"(?<!\d)(?:\d[ \-]?){13,19}\d(?!\d)",
        placeholder="[CREDIT_CARD]",
        luhn_check=True,
    ),
    PIIRule(
        name="iban",
        category="IBAN",
        pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        placeholder="[IBAN]",
    ),
    PIIRule(
        name="ipv4",
        category="IP_ADDRESS",
        pattern=r"(?<!\d\.)(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})(?!\.\d)",
        placeholder="[IP]",
    ),
    PIIRule(
        name="ipv6",
        category="IP_ADDRESS",
        pattern=r"\b(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{1,4}\b",
        placeholder="[IP]",
    ),
    PIIRule(
        name="url_credential",
        category="CREDENTIAL",
        pattern=r"https?://[^/\s:@]+:[^/\s:@]+@[^\s]+",
        placeholder="[URL_WITH_CREDENTIALS]",
    ),
    PIIRule(
        name="gps_coords",
        category="LOCATION",
        # decimal degrees pair like "12.9716, 77.5946" — common for farm GPS.
        pattern=r"(?<!\d)-?(?:90|[1-8]?\d(?:\.\d{3,8}))\s*,\s*-?(?:180|1[0-7]\d|\d{1,2})(?:\.\d{3,8})(?!\d)",
        placeholder="[GPS]",
    ),
    PIIRule(
        name="aadhaar",
        category="GOV_ID_IN",
        pattern=r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)",
        placeholder="[AADHAAR]",
    ),
    PIIRule(
        name="pan",
        category="GOV_ID_IN",
        pattern=r"\b[A-Z]{5}\d{4}[A-Z]\b",
        placeholder="[PAN]",
    ),
    PIIRule(
        name="gstin",
        category="GOV_ID_IN",
        pattern=r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b",
        placeholder="[GSTIN]",
    ),
    PIIRule(
        name="aws_access_key",
        category="CREDENTIAL",
        pattern=r"\b(?:AKIA|ASIA|AGPA)[0-9A-Z]{16}\b",
        placeholder="[AWS_KEY]",
        flags=0,  # case-sensitive
    ),
    PIIRule(
        name="generic_secret",
        category="CREDENTIAL",
        # Heuristic: long base64-ish strings prefixed with sk-/api-/Bearer
        pattern=r"\b(?:sk|api|key|token|bearer)[-_][A-Za-z0-9_\-]{16,}\b",
        placeholder="[SECRET]",
    ),
    PIIRule(
        name="ssn_us",
        category="GOV_ID_US",
        pattern=r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
        placeholder="[SSN]",
    ),
)


def detect_all(text: str, rules: tuple[PIIRule, ...] = BUILTIN_RULES) -> list[PIIDetection]:
    """Return every detection in ``text``, sorted left-to-right.

    Overlapping detections are kept; the redactor resolves them by preferring
    the leftmost-longest match.
    """
    detections: list[PIIDetection] = []
    for rule in rules:
        for m in rule.compiled().finditer(text):
            if rule.luhn_check and not _luhn_valid(m.group(0)):
                continue
            detections.append(
                PIIDetection(
                    rule=rule.name,
                    category=rule.category,
                    start=m.start(),
                    end=m.end(),
                    text=m.group(0),
                )
            )
    detections.sort(key=lambda d: (d.start, -d.end))
    return detections


def load_rules(path: str | Path) -> tuple[PIIRule, ...]:
    """Load rules from a YAML file. Schema:

    .. code-block:: yaml

        rules:
          - name: emp_id
            category: INTERNAL_ID
            pattern: 'EMP-\\d{6}'
            placeholder: '[EMP_ID]'
            flags: 0

    The returned tuple is *appended* to the built-ins by callers that want
    overrides; pass ``include_builtins: false`` at the top level to opt out.
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw_rules = data.get("rules", []) or []
    out: list[PIIRule] = []
    for r in raw_rules:
        out.append(
            PIIRule(
                name=str(r["name"]),
                category=str(r.get("category", r["name"].upper())),
                pattern=str(r["pattern"]),
                placeholder=str(r.get("placeholder", f"[{r['name'].upper()}]")),
                flags=int(r.get("flags", re.IGNORECASE)),
                luhn_check=bool(r.get("luhn_check", False)),
                description=str(r.get("description", "")),
            )
        )
    if data.get("include_builtins", True):
        return tuple(BUILTIN_RULES) + tuple(out)
    return tuple(out)


def rules_to_dict(rules: tuple[PIIRule, ...]) -> list[dict[str, Any]]:
    """Serialise rules for audit and report output."""
    return [
        {
            "name": r.name,
            "category": r.category,
            "placeholder": r.placeholder,
            "luhn_check": r.luhn_check,
            "description": r.description,
        }
        for r in rules
    ]

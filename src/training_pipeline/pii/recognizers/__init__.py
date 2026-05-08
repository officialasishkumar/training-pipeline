"""Custom recognizers for Indian identifiers and structured-field PII.

These extend the regex-only built-ins with proper checksum validation
(Verhoeff for Aadhaar, alphabetic-checksum for PAN) so we don't flag
random 12-digit phone numbers or 5L4D1L strings as government IDs.
"""

from training_pipeline.pii.recognizers.indian_ids import (
    AADHAAR_RULE,
    DRIVING_LICENSE_RULE,
    INDIAN_ID_RULES,
    INDIAN_MOBILE_RULE,
    PAN_RULE,
    VOTER_ID_RULE,
    is_valid_aadhaar,
    is_valid_pan,
)

__all__ = [
    "AADHAAR_RULE",
    "DRIVING_LICENSE_RULE",
    "INDIAN_ID_RULES",
    "INDIAN_MOBILE_RULE",
    "PAN_RULE",
    "VOTER_ID_RULE",
    "is_valid_aadhaar",
    "is_valid_pan",
]

"""Deterministic identity verification: at least 3 of 5 fields must match ONE record.

The LLM extracts what the caller said; this module alone decides whether the caller
is verified. The result never says which field failed, so neither the agent nor
the caller can learn that.
"""

from dataclasses import dataclass

from backend.data import Policyholder
from backend.sop import normalize
from backend.sop.spec import IDENTITY_FIELDS, REQUIRED_IDENTITY_MATCHES, IdentityField

_NORMALIZERS = {
    IdentityField.DOB: normalize.normalize_dob,
    IdentityField.PHONE: normalize.normalize_phone,
    IdentityField.EMAIL: normalize.normalize_email,
    IdentityField.ID_LAST4: normalize.normalize_id_last4,
}


def normalize_identity_value(field: IdentityField, raw: str) -> str | None:
    """Canonical form of a caller-provided value, or None if it can't be read."""
    if field == IdentityField.FULL_NAME:
        return raw.strip() if len(normalize.name_tokens(raw)) >= 2 else None
    return _NORMALIZERS[field](raw)


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    party_id: str | None
    # True when enough fields were provided to make a decision. A failed evaluated
    # attempt counts toward the retry limit; "not enough info yet" does not.
    evaluated: bool
    missing_fields: list[IdentityField]


def _field_matches(field: IdentityField, value: str, record: Policyholder) -> bool:
    if field == IdentityField.FULL_NAME:
        return normalize.names_match(value, record.all_names)
    if field == IdentityField.DOB:
        return value == record.dob
    if field == IdentityField.PHONE:
        return value in {normalize.normalize_phone(p) for p in record.all_phones}
    if field == IdentityField.EMAIL:
        return value in {e.lower() for e in record.all_emails}
    if field == IdentityField.ID_LAST4:
        return value == record.id_last4
    return False


def verify_identity(
    provided: dict[IdentityField, str], records: list[Policyholder]
) -> VerificationResult:
    """`provided` holds already-normalized values (see normalize_identity_value)."""
    missing = [f for f in IDENTITY_FIELDS if f not in provided]

    if len(provided) < REQUIRED_IDENTITY_MATCHES:
        return VerificationResult(False, None, evaluated=False, missing_fields=missing)

    matching_records = [
        record
        for record in records
        if sum(_field_matches(f, v, record) for f, v in provided.items())
        >= REQUIRED_IDENTITY_MATCHES
    ]

    # Exactly one record must match; anything else is treated as not verified.
    if len(matching_records) == 1:
        return VerificationResult(
            True, matching_records[0].party_id, evaluated=True, missing_fields=missing
        )
    return VerificationResult(False, None, evaluated=True, missing_fields=missing)

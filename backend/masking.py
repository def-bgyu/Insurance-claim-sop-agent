"""PII masking for anything written to traces or shown in the debug panel.

Real values are used in memory for verification; only masked versions are logged.
"""

import re

from backend.sop.spec import IdentityField

_EMAIL = re.compile(r"([\w.+-])[\w.+-]*(@[\w-]+(?:\.[\w-]+)+)")
_PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?(\d{4})\b")
_ISO_DATE = re.compile(r"\b((?:19|20)\d{2})-\d{1,2}-\d{1,2}\b")
_US_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/((?:19|20)\d{2})\b")
_LAST4 = re.compile(
    r"(\b(?:ssn|social(?: security)?|national id|last (?:four|4)(?: digits)?)\b\D{0,25}?)\d{3}(\d)\b",
    re.IGNORECASE,
)


def mask_value(field: IdentityField, value: str) -> str:
    if field == IdentityField.ID_LAST4:
        return "***" + value[-1:]
    if field == IdentityField.DOB:
        return value[:4] + "-**-**"
    if field == IdentityField.PHONE:
        return "***-***-" + value[-4:]
    if field == IdentityField.EMAIL:
        return _EMAIL.sub(r"\1***\2", value)
    if field == IdentityField.FULL_NAME:
        parts = value.split()
        return " ".join([parts[0], *(p[0] + "." for p in parts[1:])]) if parts else value
    return "***"


def mask_text(text: str) -> str:
    """Best-effort masking of PII inside free text (caller messages, raw LLM output)."""
    text = _EMAIL.sub(r"\1***\2", text)
    text = _ISO_DATE.sub(r"\1-**-**", text)
    text = _US_DATE.sub(r"**/**/\1", text)
    text = _PHONE.sub(r"***-***-\1", text)
    text = _LAST4.sub(r"\1***\2", text)
    return text

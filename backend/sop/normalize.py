"""Turn whatever the caller typed into canonical values so matching can be exact.

Each normalizer returns None when the input cannot be interpreted. A None value is
treated as "not provided", never as a mismatch.
"""

import re
import unicodedata
from datetime import datetime

from dateutil import parser as date_parser

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def name_tokens(name: str) -> list[str]:
    """'Margaret A. Chen' -> ['margaret', 'a', 'chen'] (accents and punctuation removed)."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z\s'-]", " ", text).lower()
    text = re.sub(r"['-]", "", text)
    return text.split()


def names_match(given: str, record_names: list[str]) -> bool:
    """Lenient on formatting, strict on identity. Nicknames do NOT match.

    Accepts: case/punctuation/accent differences, a middle name or initial that is
    not on file, given/family order swapped, and spacing differences ("Yawen Li").
    """
    given_tokens = name_tokens(given)
    if len(given_tokens) < 2:
        return False  # a first name alone is not a full name

    for record_name in record_names:
        record_tokens = name_tokens(record_name)
        if len(record_tokens) < 2:
            continue
        if "".join(given_tokens) == "".join(record_tokens):
            return True
        if sorted(given_tokens) == sorted(record_tokens):
            return True
        # Extra or missing middle names: first and last must still agree.
        if given_tokens[0] == record_tokens[0] and given_tokens[-1] == record_tokens[-1]:
            return True
    return False


def normalize_dob(raw: str) -> str | None:
    """Any reasonable date format -> 'YYYY-MM-DD'. US month-first for ambiguous input."""
    text = raw.strip()
    digits = re.sub(r"\D", "", text)

    # Pure digit strings: YYYYMMDD or MMDDYYYY.
    if text.isdigit() and len(digits) == 8:
        for fmt in ("%Y%m%d", "%m%d%Y"):
            try:
                return datetime.strptime(digits, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    # Parse twice with different defaults: if the result changes, the input was
    # incomplete (e.g. "March 1985" has no day) and we must not guess.
    try:
        a = date_parser.parse(text, default=datetime(2000, 1, 1), fuzzy=True)
        b = date_parser.parse(text, default=datetime(2001, 2, 2), fuzzy=True)
    except (ValueError, OverflowError):
        return None
    if a != b:
        return None
    return a.date().isoformat()


def normalize_phone(raw: str) -> str | None:
    """Keep the last 10 digits (drops a +1 country code). Fewer than 10 -> None."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def normalize_email(raw: str) -> str | None:
    email = raw.strip().lower()
    return email if _EMAIL_RE.match(email) else None


def normalize_id_last4(raw: str) -> str | None:
    """Exactly four digits. A full SSN is rejected rather than truncated."""
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) == 4 else None

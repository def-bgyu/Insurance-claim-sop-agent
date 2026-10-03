"""Extractor: turn one caller message into structured signals.

The LLM *proposes* values; nothing here changes state. Two sources are merged:
  1. Deterministic regexes for strictly formatted values (email, phone, dates, IDs).
     These work even if the model fails completely.
  2. The LLM, for everything that needs language understanding (names, intent,
     case hints, emotion, scope, yes/no answers).
Any field the model gets wrong or malformed is dropped, never guessed.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from backend.llm import prompts
from backend.llm.parsing import parse_json_object
from backend.llm.provider import LLMProvider, LLMResult, Message
from backend.sop.spec import Intent


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    CONFUSED = "confused"


class YesNo(StrEnum):
    YES = "yes"
    NO = "no"


_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}
_STATUS_SYNONYMS = {
    "denied": "denied", "rejected": "denied", "declined": "denied",
    "open": "open", "pending": "open", "in progress": "open", "processing": "open",
    "closed": "closed", "paid": "closed", "settled": "closed", "completed": "closed",
}


class Extraction(BaseModel):
    """Flat on purpose: small models handle flat schemas far better than nested ones."""

    # Identity (as the caller stated it; code normalizes and verifies).
    full_name: str | None = None
    dob: str | None = None
    phone: str | None = None
    email: str | None = None
    id_last4: str | None = None
    policy_number: str | None = None
    caller_role: str | None = None  # "policyholder" | "representative"
    representative_name: str | None = None

    # Case hints (may arrive in any phase).
    case_type: str | None = None
    case_status: str | None = None
    case_month: int | None = None
    case_year: int | None = None
    case_id: str | None = None

    # Conversation signals.
    intent: Intent | None = None
    followup_topic: str | None = None
    confirms_case: YesNo | None = None
    email_consent: YesNo | None = None
    emotion: Emotion = Emotion.NEUTRAL
    refuses_to_share: bool = False
    off_topic: bool = False
    wants_human: bool = False
    wants_to_end: bool = False

    @field_validator("case_month", mode="before")
    @classmethod
    def _month(cls, v: Any):
        if not isinstance(v, str):
            return v
        text = v.strip().lower()
        if text.isdigit():
            return int(text)
        # "January", "jan", "Jan." all map to 1.
        return next(
            (num for name, num in _MONTHS.items() if len(text) >= 3 and name.startswith(text[:3])),
            None,
        )

    @field_validator("case_month")
    @classmethod
    def _month_range(cls, v: int | None):
        return v if v is None or 1 <= v <= 12 else None

    @field_validator("case_status", mode="before")
    @classmethod
    def _status(cls, v: Any):
        return _STATUS_SYNONYMS.get(str(v).strip().lower()) if v else None

    @field_validator("case_type", "caller_role", mode="before")
    @classmethod
    def _lower(cls, v: Any):
        return str(v).strip().lower() or None if v else None

    @field_validator("case_id", "policy_number", mode="before")
    @classmethod
    def _upper(cls, v: Any):
        return str(v).strip().upper() or None if v else None

    @field_validator(
        "full_name", "dob", "phone", "email", "id_last4", "representative_name",
        "followup_topic", mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v: Any):
        if v is None:
            return None
        v = str(v).strip()
        return None if v.lower() in {"", "null", "none", "n/a", "unknown"} else v

    def useful_fields(self) -> set[str]:
        """Fields carrying real information (anything not at its default)."""
        return set(self.model_dump(exclude_defaults=True))


FIELD_NAMES = list(Extraction.model_fields)


def coerce_extraction(data: dict) -> Extraction:
    """Validate leniently: a bad value drops that one field, not the whole extraction."""
    data = {k: v for k, v in data.items() if k in Extraction.model_fields}
    for _ in range(len(FIELD_NAMES)):
        try:
            return Extraction.model_validate(data)
        except ValidationError as e:
            for err in e.errors():
                data.pop(err["loc"][0], None)
    return Extraction()


# --- Deterministic pre-pass -----------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_ISO_DATE = re.compile(r"\b(19|20)\d{2}-\d{1,2}-\d{1,2}\b")
_US_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/(19|20)\d{2}\b")
_LAST4_CONTEXT = re.compile(
    r"\b(?:ssn|social(?: security)?|national id|last (?:four|4)(?: digits)?)\b\D{0,25}?(\d{4})\b",
    re.IGNORECASE,
)
_DOB_CONTEXT = re.compile(r"\b(?:dob|d\.o\.b|birth|born|birthday)\b", re.IGNORECASE)
_POLICY = re.compile(r"\bPOL-?\d+\b", re.IGNORECASE)
_CASE = re.compile(r"\bCL-?\d+\b", re.IGNORECASE)


def regex_prepass(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    if m := _EMAIL.search(text):
        found["email"] = m.group(0)
    # A date is only a DOB if the caller says so, or the message is little more than
    # the date itself (an answer to "what's your date of birth?"). Otherwise
    # "my claim from 01/12/2026" would be misread as a birth date.
    if m := _ISO_DATE.search(text) or _US_DATE.search(text):
        if _DOB_CONTEXT.search(text) or len(text.replace(m.group(0), "").strip()) < 15:
            found["dob"] = m.group(0)
    # Remove dates before looking for phones so "1985-03-15" isn't read as digits.
    without_dates = _US_DATE.sub(" ", _ISO_DATE.sub(" ", text))
    if m := _PHONE.search(without_dates):
        found["phone"] = m.group(0)
    if m := _LAST4_CONTEXT.search(text):
        found["id_last4"] = m.group(1)
    if m := _POLICY.search(text):
        found["policy_number"] = m.group(0).upper().replace("POL", "POL-").replace("--", "-")
    if m := _CASE.search(text):
        found["case_id"] = m.group(0).upper().replace("CL", "CL-").replace("--", "-")
    return found


_CASE_TYPE_WORDS = {
    "healthcare": r"health ?care|medical|doctor|hospital",
    "dental": r"dental|dentist",
    "auto": r"auto|car|vehicle",
}
_DENIED_WORDS = re.compile(r"\b(?:denied|rejected|declined)\b", re.IGNORECASE)
_CLAIM_MONTH = re.compile(
    r"\b(?:from|in|since|back in)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?"
    r"|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|correct|right|sure|please do|that's (?:it|right|the one))\b", re.I)
_NO = re.compile(r"^\s*(?:no|nope|nah|not really|no thanks)\b", re.I)
_DONE = re.compile(r"\b(?:that's all|that is all|nothing else|that's it for|i'm done|bye)\b", re.I)


def keyword_hints(text: str) -> dict[str, str]:
    """Backup case hints from plain keywords. Used only where the LLM gave nothing,
    because the LLM understands negation ("not my dental claim") and keywords don't."""
    found: dict[str, str] = {}
    types = [t for t, words in _CASE_TYPE_WORDS.items() if re.search(rf"\b(?:{words})\b", text, re.I)]
    if len(types) == 1:
        found["case_type"] = types[0]
    if _DENIED_WORDS.search(text):
        found["case_status"] = "denied"
    if m := _CLAIM_MONTH.search(text):
        found["case_month"] = m.group(1)
    # Short answers to yes/no questions. The engine reads whichever applies to its question.
    if _YES.match(text):
        found["confirms_case"] = found["email_consent"] = "yes"
    elif _NO.match(text):
        found["confirms_case"] = found["email_consent"] = "no"
    if _DONE.search(text):
        found["wants_to_end"] = True
    return found


# --- LLM extraction -------------------------------------------------------------------


@dataclass
class ExtractionOutcome:
    extraction: Extraction
    parse_layer: str  # parser layer that worked, "failed", or "llm_error"
    regex_fields: dict[str, str]
    llm: LLMResult | None


def extract(
    provider: LLMProvider,
    user_text: str,
    phase: str,
    last_agent_message: str | None,
    followup_topics: list[str],
) -> ExtractionOutcome:
    regex_fields = regex_prepass(user_text)

    system = prompts.extractor_system(phase, last_agent_message, followup_topics)
    result = provider.complete(system, [Message("user", user_text)], max_tokens=600)

    if result.error:
        parsed_data, layer = {}, "llm_error"
    else:
        parsed = parse_json_object(result.text, tag="json", field_names=FIELD_NAMES)
        parsed_data, layer = parsed.data or {}, parsed.layer

    # Priority: keyword hints < LLM < regex. Regex values are literal matches of
    # strictly formatted fields, so they win; keywords only fill gaps.
    llm_values = {k: v for k, v in parsed_data.items() if v not in (None, "", [])}
    merged = {**keyword_hints(user_text), **llm_values, **regex_fields}
    return ExtractionOutcome(coerce_extraction(merged), layer, regex_fields, result)

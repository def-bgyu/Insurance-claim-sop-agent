#Holding trace of each step in the workflow

import json
import re
from pathlib import Path

from backend.config import ROOT_DIR
from backend.sop.spec import IdentityField
from backend.sop.state import SessionState

TRACE_DIR = ROOT_DIR / "traces"


# --- PII masking ------------------------------------------------------------------------
# Real values are used in memory for verification; only masked versions are logged
# or shown in the debug panel.

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

    text = _EMAIL.sub(r"\1***\2", text)
    text = _ISO_DATE.sub(r"\1-**-**", text)
    text = _US_DATE.sub(r"**/**/\1", text)
    text = _PHONE.sub(r"***-***-\1", text)
    text = _LAST4.sub(r"\1***\2", text)
    return text


def public_snapshot(state: SessionState) -> dict:
    #what the debug panel shows
    return {
        "session_id": state.session_id,
        "phase": state.phase.value,
        "verified": state.verified,
        "caller_role": state.caller_role,
        "identity_collected": {f.value: mask_value(f, v) for f, v in state.identity.items()},
        "remembered_hints": state.hints.model_dump(exclude_defaults=True),
        "policy_number_hint": state.policy_number_hint,
        "intent": state.intent.value if state.intent else None,
        "candidate_case_ids": state.candidate_case_ids,
        "pending_question": state.pending_question,
        "selected_case_id": state.selected_case_id,
        "counters": state.counters.model_dump(),
        "email": state.email.model_dump(),
        "handoff_requested": state.handoff_requested,
        "consent": state.consent.model_dump(exclude={"party_id"}),
        "escalation_reason": state.escalation_reason,
    }


def mask_fields(fields: dict) -> dict:
    #Mask identities, dont reveal personal info
    masked = {}
    for key, value in fields.items():
        if key in IdentityField._value2member_map_ and isinstance(value, str):
            masked[key] = mask_value(IdentityField(key), value)
        elif isinstance(value, str):
            masked[key] = mask_text(value)
        else:
            masked[key] = value
    return masked


class TraceWriter:
    def __init__(self, directory: Path = TRACE_DIR, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled

    def write(self, session_id: str, record: dict) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"{session_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

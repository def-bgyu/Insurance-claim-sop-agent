#Holding trace of each step in the workflow

import json
import re
from pathlib import Path

from backend.config import ROOT_DIR
from backend.sop.normalize import FULL_SSN
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
# A written-out date right after a DOB cue: "born 15 March 1985", "DOB is March 15, 1985".
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_WORD_DATE = rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?),?\s+(?:19|20)\d{{2}}"
_DOB_WORD_DATE = re.compile(
    rf"(\b(?:born|birth|dob|d\.o\.b|birthday)\b\D{{0,15}}?){_WORD_DATE}", re.IGNORECASE
)
# Identity values inside structured model output: "id_last4": "4472" or <dob>…</dob>.
_STRUCTURED = re.compile(
    r'("(?P<jkey>full_name|dob|phone|email|id_last4)"\s*:\s*")(?P<jval>[^"]*)(")'
    r"|(<(?P<tkey>full_name|dob|phone|email|id_last4)>)(?P<tval>[^<]*)(</(?P=tkey)>)"
)
# Any date-shaped text, to find the caller's DOB written in another format.
_ANY_DATE = re.compile(
    rf"\b(?:(?:19|20)\d{{2}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}/\d{{1,2}}/(?:19|20)\d{{2}}|{_WORD_DATE})\b",
    re.IGNORECASE,
)


def mask_value(field: IdentityField, value: str) -> str:
    if field == IdentityField.ID_LAST4:
        return "***" + value[-1:]
    if field == IdentityField.DOB:
        # Keep the year only for ISO dates; anything else ("15 March 1985") is fully hidden.
        return value[:4] + "-**-**" if re.match(r"^\d{4}-", value) else "****-**-**"
    if field == IdentityField.PHONE:
        return "***-***-" + value[-4:]
    if field == IdentityField.EMAIL:
        return _EMAIL.sub(r"\1***\2", value)
    if field == IdentityField.FULL_NAME:
        parts = value.split()
        return " ".join([parts[0], *(p[0] + "." for p in parts[1:])]) if parts else value
    return "***"


def _mask_structured(match: re.Match) -> str:
    key = match.group("jkey") or match.group("tkey")
    value = match.group("jval") if match.group("jkey") else match.group("tval")
    masked = mask_value(IdentityField(key), value) if value else value
    if match.group("jkey"):
        return f"{match.group(1)}{masked}{match.group(4)}"
    return f"{match.group(5)}{masked}{match.group(8)}"


def mask_text(text: str) -> str:
    """Pattern-based masking for free text (caller messages, raw model output)."""
    text = _STRUCTURED.sub(_mask_structured, text)
    text = FULL_SSN.sub("***-**-****", text)
    text = _EMAIL.sub(r"\1***\2", text)
    text = _DOB_WORD_DATE.sub(r"\1****-**-**", text)
    text = _ISO_DATE.sub(r"\1-**-**", text)
    text = _US_DATE.sub(r"**/**/\1", text)
    text = _PHONE.sub(r"***-***-\1", text)
    text = _LAST4.sub(r"\1***\2", text)
    return text


def redact(text: str, known: dict[IdentityField, set[str]]) -> str:
    """Mask by value: every identity value seen this turn is replaced wherever it
    appears, in any format, then pattern masking catches the rest. Patterns alone
    miss phrasings nobody anticipated; values don't."""
    from backend.sop.normalize import normalize_dob  # local: avoids an import cycle

    for field, values in known.items():
        for value in sorted(values, key=len, reverse=True):
            if len(value) >= 4:
                text = re.sub(re.escape(value), mask_value(field, value), text, flags=re.IGNORECASE)
    dobs = {normalize_dob(v) for v in known.get(IdentityField.DOB, set())} - {None}
    if dobs:
        text = _ANY_DATE.sub(
            lambda m: "****-**-**" if normalize_dob(m.group(0)) in dobs else m.group(0), text
        )
    return mask_text(text)


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

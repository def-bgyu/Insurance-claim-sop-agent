"""Per-turn execution traces (JSONL, one file per session) and the public state snapshot.

A trace records every step of a turn: extraction (and which parser layer worked),
state changes, the policy decision and its reasons, the facts the responder was
allowed to see, and whether the LLM reply or the fallback was sent. PII is masked.
"""

import json
from pathlib import Path

from backend.config import ROOT_DIR
from backend.masking import mask_text, mask_value
from backend.sop.state import SessionState

TRACE_DIR = ROOT_DIR / "traces"


def public_snapshot(state: SessionState) -> dict:
    """What the debug panel shows. Identity values are masked."""
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
        "escalation_reason": state.escalation_reason,
    }


def mask_fields(fields: dict) -> dict:
    """Mask identity-like values inside an extraction dict."""
    from backend.sop.spec import IdentityField

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

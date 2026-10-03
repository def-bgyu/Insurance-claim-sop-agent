"""Per-conversation state. Lives on the server; the browser only ever sees a snapshot.

The LLM proposes values (extracted fields, intent hints); code decides what gets
written here and which phase we are in.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from backend.sop.spec import ALLOWED_TRANSITIONS, TERMINAL_PHASES, IdentityField, Intent, Phase


class CaseHints(BaseModel):
    """Anything the caller said about their case, remembered across phases.

    Example: during VERIFY_ID the caller says "my denied healthcare claim from
    January" -> case_type="healthcare", status="denied", month=1. We stay in
    VERIFY_ID but use these hints once the caller is verified.
    """

    case_type: str | None = None  # healthcare | dental | auto | ...
    status: str | None = None  # denied | open | closed | ...
    month: int | None = None
    year: int | None = None
    case_id: str | None = None
    notes: list[str] = Field(default_factory=list)  # free-text details worth keeping


class EmailState(BaseModel):
    offered: bool = False
    preview_shown: bool = False
    consent: bool | None = None  # None = not answered yet


class Counters(BaseModel):
    off_topic: int = 0
    frustration: int = 0
    verification_failures: int = 0


class DiscussedItem(BaseModel):
    """One answered question, used to build the email summary."""

    case_id: str
    intent: Intent
    facts: list[str]
    next_steps: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    role: str  # "user" | "agent"
    text: str
    phase: Phase
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    phase: Phase = Phase.VERIFY_ID

    # Identity: normalized values the caller has given us (never echoed back raw).
    identity: dict[IdentityField, str] = Field(default_factory=dict)
    verified_party_id: str | None = None
    caller_role: str = "policyholder"  # "policyholder" | "representative"
    representative_name: str | None = None

    # Memory that survives phase boundaries.
    hints: CaseHints = Field(default_factory=CaseHints)
    policy_number_hint: str | None = None  # a lookup hint only; never counts toward ID
    intent: Intent | None = None

    # Case selection.
    candidate_case_ids: list[str] = Field(default_factory=list)
    selected_case_id: str | None = None

    # The question the agent's last message ended with (see plan.Pending). Every
    # yes/no answer is interpreted against this and nothing else.
    pending_question: str | None = None

    discussed: list[DiscussedItem] = Field(default_factory=list)
    handoff_requested: bool = False  # caller wants a human after their case was handled

    counters: Counters = Field(default_factory=Counters)
    email: EmailState = Field(default_factory=EmailState)
    escalation_reason: str | None = None

    transcript: list[Turn] = Field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.verified_party_id is not None

    @property
    def is_over(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def transition(self, to: Phase) -> None:
        """The only way to change phase. Illegal jumps raise instead of silently passing."""
        if to == self.phase:
            return
        if to not in ALLOWED_TRANSITIONS[self.phase]:
            raise ValueError(f"Illegal SOP transition {self.phase} -> {to}")
        # Hard gate: nothing past VERIFY_ID without a verified identity.
        if to != Phase.ESCALATED and not self.verified:
            raise ValueError(f"Cannot enter {to} before identity is verified")
        self.phase = to

    def escalate(self, reason: str) -> None:
        self.escalation_reason = reason
        self.transition(Phase.ESCALATED)

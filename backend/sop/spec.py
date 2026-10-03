"""The SOP definition: phases, allowed transitions, identity fields, intents and limits.

This module is the single source of truth for the workflow. Code reads it to decide
what is allowed; the LLM never decides phase changes on its own.
"""

from enum import StrEnum


class Phase(StrEnum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    ESCALATED = "ESCALATED"  # handed off to a human; terminal
    ENDED = "ENDED"  # conversation finished normally; terminal


# Phases can only move along these edges. Any phase may escalate to a human.
ALLOWED_TRANSITIONS: dict[Phase, set[Phase]] = {
    # ENDED: a representative without consent declines a human transfer.
    Phase.VERIFY_ID: {Phase.RESOLVE_INTENT, Phase.ESCALATED, Phase.ENDED},
    # POST_PROCESS: a claim was already discussed and the caller accepts a human transfer
    # (email summary first). ENDED: no claims on file and the caller is done.
    Phase.RESOLVE_INTENT: {Phase.PROCESS_CASE, Phase.POST_PROCESS, Phase.ESCALATED, Phase.ENDED},
    # A caller may ask about a second claim, which sends us back to RESOLVE_INTENT.
    Phase.PROCESS_CASE: {Phase.POST_PROCESS, Phase.RESOLVE_INTENT, Phase.ESCALATED},
    Phase.POST_PROCESS: {Phase.ENDED, Phase.ESCALATED},
    Phase.ESCALATED: set(),
    Phase.ENDED: set(),
}

TERMINAL_PHASES = {Phase.ESCALATED, Phase.ENDED}


class IdentityField(StrEnum):
    FULL_NAME = "full_name"
    DOB = "dob"
    PHONE = "phone"
    EMAIL = "email"
    ID_LAST4 = "id_last4"  # SSN last 4, or national ID last 4 depending on the record


# Order in which missing fields are requested from the caller.
IDENTITY_FIELDS: list[IdentityField] = [
    IdentityField.FULL_NAME,
    IdentityField.DOB,
    IdentityField.ID_LAST4,
    IdentityField.PHONE,
    IdentityField.EMAIL,
]

IDENTITY_FIELD_LABELS: dict[IdentityField, str] = {
    IdentityField.FULL_NAME: "full legal name",
    IdentityField.DOB: "date of birth",
    IdentityField.PHONE: "phone number on file",
    IdentityField.EMAIL: "email address on file",
    IdentityField.ID_LAST4: "last four digits of your SSN or national ID",
}

REQUIRED_IDENTITY_MATCHES = 3


class Intent(StrEnum):
    """What the caller wants. Names align with `intent_hints` in the guidance fixture."""

    STATUS_INQUIRY = "status_inquiry"
    DENIAL_QUESTION = "denial_question"
    PAYMENT_QUESTION = "payment_question"
    DOCUMENT_SUBMISSION = "document_submission"
    NEXT_STEPS = "next_steps"
    GENERAL_CLAIM_QUESTION = "general_claim_question"
    SPEAK_TO_HUMAN = "speak_to_human"
    UNKNOWN = "unknown"


INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.STATUS_INQUIRY: "Where is my claim / what is its current status.",
    Intent.DENIAL_QUESTION: "Why was my claim denied / what does the denial mean.",
    Intent.PAYMENT_QUESTION: "I wasn't paid, or was paid less than expected / payment amounts.",
    Intent.DOCUMENT_SUBMISSION: "Which documents are needed, how/where/when to send them, format.",
    Intent.NEXT_STEPS: "What should I do now / how do I fix this / appeal.",
    Intent.GENERAL_CLAIM_QUESTION: "Any other question about a specific claim.",
    Intent.SPEAK_TO_HUMAN: "Caller explicitly asks to be transferred to a human (not a question about why).",
    Intent.UNKNOWN: "Not enough information to tell what the caller wants.",
}


class Limits:
    """After this many occurrences the agent hands the caller to a human."""

    OFF_TOPIC = 3
    FRUSTRATION = 3
    VERIFICATION_FAILURES = 3

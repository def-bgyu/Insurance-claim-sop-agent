"""ResponsePlan: what code decided this turn, handed to the responder to phrase.

`facts` is the ONLY claim information the responder may see. `fallback_text` is a
complete, correct reply body used whenever the LLM fails or breaks a grounding
rule, so every turn has a safe answer even with no working model.

`closing_question` is appended by code, word for word, after the reply body. The
question the caller must answer is therefore always the one the state expects,
and `pending_question` tells the next turn how to read a "yes" or "no".
"""

from dataclasses import dataclass, field


class Pending:
    """What the agent's closing question asked; how the next yes/no is interpreted."""

    CONFIRM_CLAIM = "confirm_claim"  # "Just to confirm, are you calling about claim X?"
    CHOOSE_CLAIM = "choose_claim"  # "Which one are you calling about?"
    ANYTHING_ELSE = "anything_else"  # "Is there anything else I can help you with?"
    OFFER_HUMAN = "offer_human"  # "Would you like me to connect you with a human...?"
    OFFER_EMAIL = "offer_email"  # "Would you like me to email you a summary...?"
    CONSENT = "awaiting_consent"  # waiting for the policyholder to approve a representative


@dataclass
class ResponsePlan:
    action: str  # short machine-readable label, e.g. "ask_identity", "answer_case"
    directive: str  # instruction to the responder for this turn
    fallback_text: str  # reply body if the LLM can't be used (no closing question)
    facts: list[str] = field(default_factory=list)
    closing_question: str | None = None
    pending_question: str | None = None
    reasons: list[str] = field(default_factory=list)  # why code chose this (for traces)
    use_llm: bool = True  # False -> send fallback_text as-is (e.g. after the call ended)

    # Names (set by the engine on every turn). Before verification the caller is not
    # addressed by name; after, only by the verified customer's name from the record.
    verified_name: str | None = None  # full name on record
    address_as: str | None = None  # e.g. "Margaret" (or "David" for her representative)
    representative: str | None = None  # e.g. "David Chen (son)" when a rep is calling
    forbidden_names: list[str] = field(default_factory=list)  # names the reply must not use

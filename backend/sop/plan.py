"""ResponsePlan: what code decided this turn, handed to the responder to phrase.

`facts` is the ONLY claim information the responder may see. `fallback_text` is a
complete, correct reply used whenever the LLM fails or breaks a grounding rule, so
every turn has a safe answer even with no working model.
"""

from dataclasses import dataclass, field


@dataclass
class ResponsePlan:
    action: str  # short machine-readable label, e.g. "ask_identity", "answer_case"
    directive: str  # instruction to the responder for this turn
    fallback_text: str
    facts: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # why code chose this (for traces)
    use_llm: bool = True  # False -> send fallback_text as-is (e.g. after the call ended)

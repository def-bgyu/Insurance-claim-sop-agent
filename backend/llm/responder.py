"""Responder: phrase the code-chosen ResponsePlan as a natural reply.

After generation, deterministic guards check the reply:
  - grounding: every claim ID, dollar amount, and date must come from the facts;
  - actions: it may not claim a transfer or email the workflow didn't perform.
Otherwise the plan's fallback text is sent instead. A model can make the reply
sound worse, but it cannot make it say something ungrounded or untrue.
"""

import re
from dataclasses import dataclass

from dateutil import parser as date_parser

from backend.llm import prompts
from backend.llm.provider import LLMProvider, LLMResult, Message
from backend.llm.sanitize import clean_reply
from backend.sop.plan import ResponsePlan
from backend.sop.state import SessionState

HISTORY_TURNS = 10

_CLAIM_ID = re.compile(r"\bCL-\d+\b", re.IGNORECASE)
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_DATE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
)


@dataclass
class ResponseOutcome:
    text: str
    source: str  # "llm" | "fallback"
    guard_violations: list[str]
    llm: LLMResult | None
    retried: bool = False  # a second attempt was made after a wrong name


def _money_values(text: str) -> set[float]:
    return {float(re.sub(r"[^\d.]", "", m)) for m in _MONEY.findall(text)}


def _date_values(text: str) -> set[str]:
    values = set()
    for m in _DATE.findall(text):
        try:
            values.add(date_parser.parse(m).date().isoformat())
        except (ValueError, OverflowError):
            values.add(m)
    return values


# Claims that an action is happening ("I'm connecting you", "I've sent the summary").
# "Would you like me to connect you?" is an offer, not a claim, and doesn't match.
_ACTION_CLAIM = re.compile(
    r"\b(?:transferr?ing you|connecting you|"
    r"(?:i'm|i am|i'll|i will|let me)\s+(?:now\s+|go ahead and\s+)?(?:transfer|connect|put you through)|"
    r"(?:i've|i have)\s+(?:sent|emailed)|sending (?:you )?(?:the|an|your) (?:email|summary))",
    re.IGNORECASE,
)


def action_violations(reply: str, plan: ResponsePlan) -> list[str]:
    """The model may only describe an action if the code's own reply for this turn does."""
    if _ACTION_CLAIM.search(reply) and not _ACTION_CLAIM.search(plan.fallback_text):
        return ["claims an action the workflow did not take"]
    return []


def grounding_violations(reply: str, plan: ResponsePlan) -> list[str]:
    allowed = " ".join([*plan.facts, plan.fallback_text, plan.closing_question or ""])
    allowed_ids = {c.upper() for c in _CLAIM_ID.findall(allowed)}
    allowed_money = _money_values(allowed)
    allowed_dates = _date_values(allowed)

    violations = []
    for claim_id in _CLAIM_ID.findall(reply):
        if claim_id.upper() not in allowed_ids:
            violations.append(f"ungrounded claim id {claim_id}")
    for amount in _money_values(reply) - allowed_money:
        violations.append(f"ungrounded amount {amount:.2f}")
    for day in _date_values(reply) - allowed_dates:
        violations.append(f"ungrounded date {day}")
    return violations


def _history(state: SessionState) -> list[Message]:
    turns = state.transcript[-HISTORY_TURNS:]
    messages = [Message("user" if t.role == "user" else "assistant", t.text) for t in turns]
    # The API expects the conversation to start with the caller.
    if messages and messages[0].role == "assistant":
        messages.insert(0, Message("user", "(call connected)"))
    return messages


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_trailing_questions(text: str) -> str:
    """Drop questions the model added at the end; code appends the real one."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    while sentences and sentences[-1].rstrip().endswith("?"):
        sentences.pop()
    return " ".join(sentences).strip()


def _finish(body: str, plan: ResponsePlan) -> str:
    if not plan.closing_question:
        return body
    return f"{body} {plan.closing_question}".strip()


def name_violations(reply: str, plan: ResponsePlan) -> list[str]:
    """Names the caller mentioned that aren't the verified customer's (or any name,
    before verification). Matched case-sensitively as capitalized words."""
    return [
        f"wrong name {name}"
        for name in plan.forbidden_names
        if re.search(rf"\b{re.escape(name)}\b", reply)
    ]


def _generate(
    provider: LLMProvider, state: SessionState, plan: ResponsePlan, directive: str
) -> tuple[str, LLMResult, list[str]]:
    """One model attempt: (cleaned body, raw result, violations)."""
    system = prompts.responder_system(directive, plan.facts, plan.verified_name, plan.address_as)
    result = provider.complete(system, _history(state), max_tokens=500)
    if result.error or not result.text.strip():
        return "", result, ["llm_error"]
    # Questions are code-owned: any question the model adds at the end is removed, whether
    # or not the plan has its own closing question to append.
    body = strip_trailing_questions(clean_reply(result.text))
    if not body:
        return "", result, ["empty"]
    violations = (
        grounding_violations(body, plan) + action_violations(body, plan) + name_violations(body, plan)
    )
    return body, result, violations


def respond(provider: LLMProvider, state: SessionState, plan: ResponsePlan) -> ResponseOutcome:
    fallback = _finish(plan.fallback_text, plan)
    if not plan.use_llm:
        return ResponseOutcome(fallback, "fallback", [], None)

    directive = plan.directive
    if plan.closing_question:
        directive += (
            "\nDo NOT end with a question. The system will append this exact question after "
            f'your message: "{plan.closing_question}"'
        )
    body, result, violations = _generate(provider, state, plan, directive)

    # A wrong name is the one mistake worth a second try: the rest of the reply is
    # usually fine, and the fallback would lose its natural tone.
    wrong_names = [v for v in violations if v.startswith("wrong name")]
    if wrong_names and len(wrong_names) == len(violations):
        fix = (
            f"address the caller as {plan.address_as} or without a name"
            if plan.address_as else "do not address the caller by any name"
        )
        retry_directive = (
            f"{directive}\nIMPORTANT: your previous draft used a name that is not the verified "
            f"customer's ({', '.join(v.removeprefix('wrong name ') for v in wrong_names)}). "
            f"Write the reply again and {fix}."
        )
        body, result, violations = _generate(provider, state, plan, retry_directive)
        if violations:
            return ResponseOutcome(fallback, "fallback", wrong_names + violations, result, retried=True)
        return ResponseOutcome(_finish(body, plan), "llm", wrong_names, result, retried=True)

    if violations:
        return ResponseOutcome(fallback, "fallback", violations, result)
    return ResponseOutcome(_finish(body, plan), "llm", [], result)

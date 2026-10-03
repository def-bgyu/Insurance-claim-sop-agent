"""Responder: phrase the code-chosen ResponsePlan as a natural reply.

After generation, a deterministic grounding guard checks the reply: every claim ID,
dollar amount, and date it mentions must come from the plan's facts. Otherwise the
plan's fallback text is sent instead. A model can make the reply sound worse, but
it cannot make it say something ungrounded.
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
    system = prompts.responder_system(directive, plan.facts)
    result = provider.complete(system, _history(state), max_tokens=500)
    if result.error or not result.text.strip():
        return ResponseOutcome(fallback, "fallback", ["llm_error"], result)

    body = clean_reply(result.text)
    if plan.closing_question:
        body = strip_trailing_questions(body)
    violations = grounding_violations(body, plan)
    if violations or not body:
        return ResponseOutcome(fallback, "fallback", violations or ["empty"], result)
    return ResponseOutcome(_finish(body, plan), "llm", [], result)

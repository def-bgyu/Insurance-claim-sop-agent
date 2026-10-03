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
    r"(?:i've|i have)\s+(?:sent|emailed)|sending (?:you )?(?:the|an|your) (?:email|summary)|"
    # Promises to send, not only past-tense claims (live test: "I'll go ahead and send
    # the summary…" before the caller had agreed).
    r"(?:i'll|i will|i'm going to|let me|i can go ahead and)\s+(?:now\s+|go ahead and\s+)?"
    r"(?:send|email|forward)\b|(?:will|is going to) be (?:sent|emailed)|on its way)",
    re.IGNORECASE,
)


def action_violations(reply: str, plan: ResponsePlan) -> list[str]:
    """The model may only describe an action if the code's own reply for this turn does."""
    if _ACTION_CLAIM.search(reply) and not _ACTION_CLAIM.search(plan.fallback_text):
        return ["claims an action the workflow did not take"]
    return []


def _compact_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def grounding_violations(reply: str, plan: ResponsePlan) -> list[str]:
    # Claim IDs are recognized in whatever format the loaded data uses (CL-2048,
    # CASE-A7X9…), so an invented ID in a non-CL format is caught too.
    id_re = plan.claim_id_pattern or _CLAIM_ID
    allowed = " ".join([*plan.facts, plan.fallback_text, plan.closing_question or ""])
    allowed_ids = {_compact_id(c) for c in id_re.findall(allowed)}
    allowed_money = _money_values(allowed)
    allowed_dates = _date_values(allowed)

    violations = []
    for claim_id in id_re.findall(reply):
        if _compact_id(claim_id) not in allowed_ids:
            violations.append(f"ungrounded claim id {claim_id}")
    for amount in _money_values(reply) - allowed_money:
        violations.append(f"ungrounded amount {amount:.2f}")
    for day in _date_values(reply) - allowed_dates:
        violations.append(f"ungrounded date {day}")
    return violations


def _history(state: SessionState) -> list[Message]:
    turns = state.transcript[-HISTORY_TURNS:]
    messages = []
    for t in turns:
        if t.role == "agent":
            messages.append(Message("assistant", t.text))
        elif t.role == "event":  # something outside the chat, e.g. a consent decision
            messages.append(Message("user", f"[System note, not said by the caller: {t.text}]"))
        else:
            messages.append(Message("user", t.text))
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


# Commenting on the caller's feelings. Only allowed when the extractor detected an
# emotion this turn; live testing showed the model adding "I understand your
# frustration" to calm, neutral messages.
_EMOTION_TALK = re.compile(
    r"\b(?:frustrat\w*|upset\w*|i hear you|how you(?:'re| are)? feel\w*|your feelings|"
    r"sorry you(?:'re| are) feeling|you(?:'re| are| sound| seem)\s+(?:angry|anxious|worried|stressed))\b",
    re.IGNORECASE,
)


def emotion_violations(reply: str, plan: ResponsePlan) -> list[str]:
    violations = []
    if not plan.acknowledge_emotion and _EMOTION_TALK.search(reply):
        violations.append("unprompted emotion talk")
    if plan.require_apology and not re.search(r"\b(?:sorry|apologi[sz]e|apologies)\b", reply, re.I):
        violations.append("missing apology")
    return violations


def address_violations(reply: str, plan: ResponsePlan) -> list[str]:
    """Using a name to address the caller: "Thanks, Margaret." / "Margaret, I…"."""
    violations = []
    for name in plan.no_address_names:
        n = re.escape(name)
        vocative = (
            rf",\s*{n}\s*[,.!?]|(?:^|[.!?]\s+){n},|"
            rf"\b(?:hi|hello|thanks|thank you|dear|okay|sure|sorry)\b,?\s+{n}\b"
        )
        if re.search(vocative, reply, re.IGNORECASE):
            violations.append(f"addresses caller as {name}")
    return violations


# Mistakes worth one regeneration: the rest of the reply is usually fine, and the
# fallback would lose the natural tone. Everything else goes straight to the fallback.
def _retry_note(violations: list[str], plan: ResponsePlan) -> str | None:
    notes = []
    fix = (
        f"address the caller as {plan.address_as} or without a name"
        if plan.address_as else "do not address the caller by any name"
    )
    names = [v.removeprefix("wrong name ") for v in violations if v.startswith("wrong name")]
    if names:
        notes.append(
            f"Your previous draft used a name that is not the verified customer's "
            f"({', '.join(names)}); {fix}."
        )
    addressed = [v.removeprefix("addresses caller as ") for v in violations if v.startswith("addresses caller as ")]
    if addressed:
        notes.append(
            f"Your previous draft addressed the caller as {', '.join(addressed)}, which is not "
            f"who is calling; {fix}."
        )
    if "unprompted emotion talk" in violations:
        notes.append(
            "Your previous draft commented on the caller's feelings; the caller has not "
            "expressed any, so do not mention feelings or frustration."
        )
    if "missing apology" in violations:
        notes.append("Begin with a brief, sincere apology for their experience.")
    retryable = (
        len(names) + len(addressed)
        + ("unprompted emotion talk" in violations) + ("missing apology" in violations)
    )
    if not notes or retryable != len(violations):
        return None
    return "IMPORTANT: " + " ".join(notes) + " Write the reply again."


def _generate(
    provider: LLMProvider, state: SessionState, plan: ResponsePlan, directive: str
) -> tuple[str, LLMResult, list[str]]:
    """One model attempt: (cleaned body, raw result, violations)."""
    system = prompts.responder_system(
        directive, plan.facts, plan.verified_name, plan.address_as, plan.representative
    )
    result = provider.complete(system, _history(state), max_tokens=500)
    if result.error or not result.text.strip():
        return "", result, ["llm_error"]
    # Questions are code-owned: any question the model adds at the end is removed, whether
    # or not the plan has its own closing question to append.
    body = strip_trailing_questions(clean_reply(result.text))
    if not body:
        return "", result, ["empty"]
    violations = (
        grounding_violations(body, plan) + action_violations(body, plan)
        + name_violations(body, plan) + address_violations(body, plan)
        + emotion_violations(body, plan)
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

    if note := _retry_note(violations, plan):
        first = violations
        body, result, violations = _generate(provider, state, plan, f"{directive}\n{note}")
        if violations:
            return ResponseOutcome(fallback, "fallback", first + violations, result, retried=True)
        return ResponseOutcome(_finish(body, plan), "llm", first, result, retried=True)

    if violations:
        return ResponseOutcome(fallback, "fallback", violations, result)
    return ResponseOutcome(_finish(body, plan), "llm", [], result)

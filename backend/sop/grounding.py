"""PROCESS_CASE helper: build the exact facts the agent may state for a question.

Everything here comes from claims.json, claim_schema.json and
required_document_guideline.json. The responder can rephrase these facts but
cannot add to them.
"""

from dataclasses import dataclass, field
from datetime import date

from backend.data import Claim, FollowupTopic, InsuranceData
from backend.sop.spec import Intent

HUMAN_REVIEW_STEP = "Speak with a human claims representative to review the available options."


@dataclass
class GroundedAnswer:
    facts: list[str]
    next_steps: list[str] = field(default_factory=list)
    needs_human: bool = False
    topics_used: list[str] = field(default_factory=list)


def _money(amount: str) -> str:
    return f"${float(amount):,.2f}"


def _day(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def deadline_passed(claim: Claim, today: date) -> bool:
    return claim.appeal_deadline is not None and claim.appeal_deadline < today


# Phrases for topics the guideline file gives no trigger phrases for. "What if I can't
# get the documents?" must reach the alternatives guidance (live test: a two-part
# question lost this half).
_EXTRA_PHRASES = {
    "missing_required_material_alternatives": (
        "can't get", "cannot get", "can not get", "can't obtain", "cannot obtain",
        "unable to get", "unable to obtain", "don't have the", "don't have them",
        "don't have it", "do not have the", "what if i can't", "what if i cannot",
        "alternative", "lost the", "no longer have",
    ),
}


def match_followup_topics(
    data: InsuranceData, claim: Claim, user_text: str, llm_topics: list[str] | None
) -> list[FollowupTopic]:
    """Every topic the caller asked about: phrases from the guideline file, extra
    phrases for topics it has none for, and the LLM's choices. A message can ask
    several things ("where do I send them, and what if I can't get them?")."""
    topics = [t for t in data.followup_topics() if claim.documents_needed or not t.requires_documents]
    text = user_text.lower()
    chosen = set(llm_topics or [])
    return [
        t for t in topics
        if any(p in text for p in (*t.match_any, *_EXTRA_PHRASES.get(t.topic, ()))) or t.topic in chosen
    ]


def _fill(template: str, claim: Claim, data: InsuranceData) -> str:
    settings = data.followup_settings()
    return template.format(
        case_id=claim.case_id,
        documents=_join(claim.documents_needed),
        average_processing_time_after_submission=settings[
            "average_processing_time_after_submission"
        ],
    )


def _base_facts(claim: Claim) -> list[str]:
    return [
        f"Claim {claim.case_id} is a {claim.case_type} claim filed on {_day(claim.created_at)}.",
        f"Current status: {claim.status}. Summary: {claim.summary}.",
    ]


def _payment_facts(claim: Claim) -> list[str]:
    facts = [f"Amount paid by the insurer so far (net pay): {_money(claim.net_pay)}."]
    if claim.status == "denied":
        facts.append("No payment was issued because the claim was denied.")
    else:
        facts.append(
            f"Expected reimbursement amount: {_money(claim.expected_reimbursement_amount)}."
        )
    facts.append(
        f"Allowed maximum amount (the most an in-network insurer pays for this service): "
        f"{_money(claim.allowed_max_amount)}."
    )
    return facts


def _denial_facts(claim: Claim, today: date) -> tuple[list[str], list[str], bool]:
    """Facts, next steps, needs_human for a denied claim."""
    facts = [f"Reason for denial: {claim.denial_reason}."]
    if claim.documents_needed:
        facts.append(f"Documents needed: {_join(claim.documents_needed)}.")
    if claim.appeal_deadline is None:
        return facts, [], False
    if deadline_passed(claim, today):
        # Decision: the claim record outranks the generic "submit within a week" guidance.
        facts.append(
            f"The appeal deadline on record was {_day(claim.appeal_deadline)}, which has "
            "already passed, so a human claims representative needs to review the options."
        )
        return facts, [HUMAN_REVIEW_STEP], True
    facts.append(f"Appeal deadline: {_day(claim.appeal_deadline)}.")
    steps = []
    if claim.documents_needed:
        steps.append(
            f"Submit the {_join(claim.documents_needed)} before {_day(claim.appeal_deadline)}."
        )
    return facts, steps, False


def build_answer(
    data: InsuranceData,
    claim: Claim,
    intent: Intent,
    user_text: str,
    llm_topics: list[str] | None,
    today: date,
) -> GroundedAnswer:
    facts = _base_facts(claim)
    next_steps: list[str] = []
    needs_human = False
    topics_used: list[str] = []

    if claim.status == "denied":
        denial, denial_steps, needs_human = _denial_facts(claim, today)
    else:
        denial, denial_steps = [], []

    if intent == Intent.PAYMENT_QUESTION:
        facts += _payment_facts(claim) + denial
        next_steps += denial_steps
    elif intent in (Intent.DENIAL_QUESTION, Intent.STATUS_INQUIRY):
        facts += denial
        next_steps += denial_steps
        if claim.status == "open":
            facts.append(
                f"Expected reimbursement amount: {_money(claim.expected_reimbursement_amount)}."
            )
        if claim.status == "closed":
            facts += _payment_facts(claim)
    else:  # document_submission, next_steps, general_claim_question, unknown
        facts += denial
        next_steps += denial_steps
        if not needs_human:
            topics = match_followup_topics(data, claim, user_text, llm_topics)
            for topic in topics:
                facts.append(_fill(topic.template, claim, data))
                topics_used.append(topic.topic)
            if "missing_required_material_alternatives" in topics_used:
                for doc in claim.documents_needed:
                    facts.append(
                        f"If the {doc} is unavailable: {data.document_alternative_guidance(doc)}"
                    )
            if claim.documents_needed and intent in (Intent.DOCUMENT_SUBMISSION, Intent.NEXT_STEPS):
                facts.append(data.default_submission_guidance())
                if guidance := data.case_type_guidance(claim.case_type):
                    facts.append(guidance)
                for doc in claim.documents_needed:
                    if doc_guidance := data.document_guidance(doc):
                        facts.append(f"About the {doc}: {doc_guidance}")
            if not topics and intent == Intent.GENERAL_CLAIM_QUESTION:
                facts.append(data.followup_fallback())

    return GroundedAnswer(facts, next_steps, needs_human, topics_used)

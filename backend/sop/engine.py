"""The SOP engine: runs one caller turn end to end.

    caller message
      -> extract      (LLM + regex: proposes values, changes nothing)
      -> remember     (store case hints / intent whatever the phase)
      -> guards       (human request, frustration, off-topic limits; any phase)
      -> phase logic  (code decides: verify, resolve claim, answer, wrap up)
      -> respond      (LLM phrases the plan; grounding guard; fallback text;
                       code appends the closing question)
      -> trace

The LLM never changes the phase, verifies anyone, or chooses which data to reveal.
Every question the caller has to answer is written by code, and every yes/no is
interpreted against `state.pending_question`, the question actually asked.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from backend import config
from backend.data import Claim, InsuranceData, get_data
from backend.llm import prompts
from backend.llm.extractor import Emotion, Extraction, YesNo, extract
from backend.llm.provider import LLMProvider
from backend.llm.responder import respond
from backend.sop import case_resolution, grounding
from backend.sop.plan import Pending, ResponsePlan
from backend.sop.spec import (
    IDENTITY_FIELD_LABELS,
    IDENTITY_FIELDS,
    REQUIRED_IDENTITY_MATCHES,
    IdentityField,
    Intent,
    Limits,
    Phase,
)
from backend.sop.state import CaseHints, DiscussedItem, SessionState, Turn
from backend.sop.normalize import normalize_email
from backend.sop.verification import normalize_identity_value, verify_identity
from backend.trace import TraceWriter, mask_fields, mask_text, mask_value, public_snapshot

_UPSET = {Emotion.FRUSTRATED, Emotion.ANGRY}

ESCALATION_MESSAGES = {
    "caller_requested_human": "Of course. I'm connecting you with a human representative now. Please hold for a moment.",
    "off_topic_limit": "I'm only able to help with insurance claim questions here, so I'm going to connect you with a human representative who can help further. Please hold for a moment.",
    "frustration_limit": "I'm really sorry this has been so frustrating. I'm connecting you with a human representative now who can help you further. Please hold for a moment.",
    "verification_failed": "I'm sorry, I wasn't able to verify your identity with the details provided. To protect your account, I'm transferring you to a human representative who can help. Please hold for a moment.",
    "representative_not_authorized": "I'm sorry, I can't share information about this policy because you're not listed as an authorized representative on it. I'm connecting you with a human representative who can explain your options. Please hold for a moment.",
}

ANYTHING_ELSE_QUESTION = "Is there anything else I can help you with?"
OFFER_HUMAN_QUESTION = (
    "Would you like me to connect you with a human representative who can help with this?"
)
DEADLINE_HUMAN_QUESTION = (
    "Would you like me to connect you with a human claims representative to review your options?"
)
REOFFER_HUMAN_QUESTION = "Would you like me to connect you with a representative?"
WHICH_CLAIM_QUESTION = "Which claim are you calling about?"
STUCK_REPEATS = 2  # the same question a 3rd time in a row -> offer a human instead


@dataclass
class TurnResult:
    reply: str
    trace: dict


def _join_or(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


def _hints_from(ex: Extraction) -> CaseHints:
    return CaseHints(
        case_type=ex.case_type, status=ex.case_status, month=ex.case_month,
        day=ex.case_day, year=ex.case_year, case_id=ex.case_id,
    )


def _describe_hints(hints: CaseHints) -> str:
    """The caller's own description of their case, e.g. 'denied healthcare claim from January'."""
    month = date(2000, hints.month, 1).strftime("%B") if hints.month else None
    day = hints.day if month else None
    parts = [hints.status, hints.case_type, "claim"]
    text = " ".join(p for p in parts if p)
    if month or hints.year:
        text += " from " + " ".join(str(p) for p in (month, day, hints.year) if p)
    if hints.case_id:
        text += f" ({hints.case_id})"
    return text


def _answer(ex: Extraction) -> YesNo | None:
    """The caller's yes/no, whichever field the extractor put it in.
    What it answers is decided by state.pending_question, not by the field name."""
    return ex.confirms_case or ex.email_consent


class SOPEngine:
    def __init__(
        self,
        provider: LLMProvider,
        data: InsuranceData | None = None,
        today: Callable[[], date] = config.today,
        tracer: TraceWriter | None = None,
    ):
        self.provider = provider
        self.data = data or get_data()
        self.today = today
        self.tracer = tracer or TraceWriter()

    # --- Public API --------------------------------------------------------------------

    def start(self) -> SessionState:
        state = SessionState()
        state.transcript.append(Turn(role="agent", text=prompts.GREETING, phase=state.phase))
        return state

    def handle(self, state: SessionState, user_text: str) -> TurnResult:
        phase_before = state.phase
        pending_before = state.pending_question
        last_agent = next((t.text for t in reversed(state.transcript) if t.role == "agent"), None)
        state.transcript.append(Turn(role="user", text=user_text, phase=state.phase))

        if state.is_over:
            plan = ResponsePlan(
                action="conversation_over",
                directive="",
                fallback_text="This conversation has ended. Please start a new chat if you need more help.",
                use_llm=False,
            )
            outcome = None
        else:
            topics = [t.topic for t in self.data.followup_topics()]
            outcome = extract(self.provider, user_text, state.phase.value, last_agent, topics)
            ex = outcome.extraction
            self._remember(state, ex)
            plan = self._guards(state, ex) or self._phase_plan(state, ex, user_text)
            plan = self._stuck_check(state, ex, plan)
            plan.acknowledge_emotion = ex.emotion != Emotion.NEUTRAL

        # Hard invariant: claim facts can never reach the responder before verification.
        if plan.facts and not state.verified:
            raise RuntimeError("SOP violation: claim facts planned for an unverified caller")

        # The question this reply ends with is the one the next yes/no answers.
        state.pending_question = plan.pending_question
        self._apply_name_policy(state, plan)

        response = respond(self.provider, state, plan)
        state.transcript.append(Turn(role="agent", text=response.text, phase=state.phase))

        trace = self._trace(state, phase_before, pending_before, user_text, outcome, plan, response)
        self.tracer.write(state.session_id, trace)
        return TurnResult(response.text, trace)

    # --- Memory ------------------------------------------------------------------------

    def _remember(self, state: SessionState, ex: Extraction) -> None:
        """Keep anything useful the caller said, even if it belongs to a later phase."""
        hints = state.hints
        for attr, value in (
            ("case_id", ex.case_id), ("case_type", ex.case_type), ("status", ex.case_status),
            ("month", ex.case_month), ("day", ex.case_day), ("year", ex.case_year),
        ):
            if value is not None:
                setattr(hints, attr, value)
        if ex.policy_number:
            state.policy_number_hint = ex.policy_number
        for name in (ex.full_name, ex.representative_name):
            if name and name not in state.names_mentioned:
                state.names_mentioned.append(name)
        if ex.intent and ex.intent not in (Intent.UNKNOWN, Intent.SPEAK_TO_HUMAN):
            state.intent = ex.intent
        if state.phase == Phase.VERIFY_ID:
            if ex.caller_role == "representative" or ex.representative_name:
                state.caller_role = "representative"
            if ex.representative_name:
                state.representative_name = ex.representative_name

    def _apply_name_policy(self, state: SessionState, plan: ResponsePlan) -> None:
        """Before verification: no names (any name typed is unconfirmed). After: only the
        name on record. Any other name the caller mentioned is forbidden in the reply."""
        mentioned = {
            token.capitalize()
            for name in state.names_mentioned
            for token in re.findall(r"[A-Za-z][A-Za-z'-]+", name)
        }
        # A representative's policyholder is confirmed from the record once their details
        # match, even while consent is still pending.
        party = state.verified_party_id or (state.consent.party_id if state.consent.status else None)
        if not party:
            plan.forbidden_names = sorted(mentioned)
            return
        holder = self.data.get_policyholder(party)
        own = {t.lower() for n in holder.all_names for t in n.split()}
        plan.verified_name = holder.name
        plan.address_as = " ".join(holder.name.split()[:-1]) or holder.name  # given name(s)
        if state.consent.status and state.consent.representative:
            rep = state.consent.representative
            own |= {t.lower() for t in rep.split()}
            plan.address_as = " ".join(rep.split()[:-1]) or rep
            plan.representative = f"{rep} ({state.consent.relationship})"
        plan.forbidden_names = sorted(t for t in mentioned if t.lower() not in own)

    # --- Cross-cutting guards ----------------------------------------------------------

    def _guards(self, state: SessionState, ex: Extraction) -> ResponsePlan | None:
        # Deterministic backup: an email that isn't on file, typed while we ask about the
        # summary email, is a request to use a different address, i.e. to change a
        # personal detail. The summary only ever goes to the address on file.
        if state.phase == Phase.POST_PROCESS and ex.email and not ex.unsupported_request:
            on_file = {e.lower() for e in self.data.get_policyholder(state.verified_party_id).all_emails}
            if (normalize_email(ex.email) or ex.email.lower()) not in on_file:
                ex.unsupported_request = "updating your email address"

        answer = _answer(ex)
        offer_open = state.pending_question == Pending.OFFER_HUMAN
        new_question = bool(ex.intent and ex.intent not in (Intent.UNKNOWN, Intent.SPEAK_TO_HUMAN)) or bool(
            ex.followup_topic
        )

        wants_human = ex.wants_human or ex.intent == Intent.SPEAK_TO_HUMAN
        if wants_human or (offer_open and answer == YesNo.YES):
            if plan := self._connect_human(state, ex):
                return plan

        # "No" to the offer, or insisting the agent do the unsupported thing itself
        # ("no, I want YOU to do it"). Saying they're done is respected, not re-offered.
        pushback = bool(ex.unsupported_request) and answer is None and state.human_offer_for_request
        declined = answer == YesNo.NO or pushback
        if offer_open and declined and not new_question and not ex.wants_to_end:
            return self._human_declined(state, ex)

        if ex.emotion in _UPSET or ex.refuses_to_share:
            state.counters.frustration += 1
            if state.counters.frustration >= Limits.FRUSTRATION:
                return self._escalate(state, "frustration_limit")

        useful = ex.useful_fields() - {"off_topic", "emotion"}
        if ex.off_topic and not useful:
            state.counters.off_topic += 1
            if state.counters.off_topic >= Limits.OFF_TOPIC:
                return self._escalate(state, "off_topic_limit")
            resume_directive, question, pending = self._resume(state)
            return ResponsePlan(
                action="decline_off_topic",
                directive=(
                    "The caller asked something unrelated to insurance claims. Do NOT answer it. "
                    "Politely say you can only help with questions about their insurance claims, "
                    f"and that you'd like to get back to {resume_directive}"
                ),
                fallback_text="I'm sorry, I can only help with questions about your insurance claims.",
                closing_question=question,
                pending_question=pending,
                reasons=[f"off_topic count {state.counters.off_topic}/{Limits.OFF_TOPIC}"],
            )

        if ex.unsupported_request and state.phase == Phase.POST_PROCESS:
            return self._email_on_file_only(state, ex)
        if ex.unsupported_request:
            return self._offer_human(
                state, ex, topic=ex.unsupported_request,
                directive=(
                    f"Say you're not able to help with {ex.unsupported_request} here, but a human "
                    "representative can."
                ),
                fallback_text=(
                    f"I'm not able to help with {ex.unsupported_request} here, but a human "
                    "representative can."
                ),
                reason=f"unsupported request: {ex.unsupported_request}",
                for_request=True,
            )
        return None

    def _email_on_file_only(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        """During wrap-up the caller asks for something we can't do, usually sending the
        summary to a different email. Personal details are never changed here; the summary
        only goes to the address on file; a human can make the change."""
        request = ex.unsupported_request
        explain = (
            "For your security, I can only send the summary to the email address on file, and "
            "I'm not able to update personal details here."
            if "email" in request else f"I'm not able to help with {request} here."
        )
        directive = (
            f"{self._tone(ex)}Explain, in your own words: \"{explain}\" Do NOT ask for another "
            "email address and do not offer to change anything."
        )
        if state.handoff_requested:
            # A transfer is already arranged: the representative can make the change.
            return ResponsePlan(
                action="email_on_file_only",
                directive=directive + " Say the representative they'll be connected to can help with that.",
                fallback_text=f"{explain} The representative you'll be connected to can help with that.",
                closing_question=self._email_question(state),
                pending_question=Pending.OFFER_EMAIL,
                reasons=[f"personal-detail request during wrap-up: {request}", "handoff already arranged"],
            )
        return self._offer_human(
            state, ex, topic=request,
            directive=directive + " Say a human representative can help with that.",
            fallback_text=f"{explain} A human representative can help with that.",
            reason=f"personal-detail request during wrap-up: {request}",
            for_request=True,
        )

    # --- Human transfer -----------------------------------------------------------------
    #
    # Offer -> "yes": connect now (or, if a claim was discussed, offer the email first)
    #       -> "no":  one gentle re-offer explaining it's the recommended route
    #                 -> "no" again: stop persuading, ask what else we can help with

    def _connect_human(self, state: SessionState, ex: Extraction) -> ResponsePlan | None:
        if state.discussed and state.phase in (Phase.PROCESS_CASE, Phase.RESOLVE_INTENT):
            # A claim was discussed: the email summary is offered before the transfer.
            state.handoff_requested = True
            state.transition(Phase.POST_PROCESS)
            return self._offer_email(state, ex)
        if state.phase == Phase.POST_PROCESS:
            # Already wrapping up: transfer right after the email question is answered.
            state.handoff_requested = True
            return None
        return self._escalate(state, "caller_requested_human")

    def _offer_human(
        self, state: SessionState, ex: Extraction, topic: str, directive: str,
        fallback_text: str, reason: str, facts: list[str] | None = None,
        question: str = OFFER_HUMAN_QUESTION, for_request: bool = False,
    ) -> ResponsePlan:
        state.human_offer_topic = topic
        state.human_offer_for_request = for_request
        state.human_offer_declines = 0
        return ResponsePlan(
            action="offer_human",
            directive=f"{self._tone(ex)}{directive}",
            fallback_text=fallback_text,
            facts=facts or [],
            closing_question=question,
            pending_question=Pending.OFFER_HUMAN,
            reasons=[reason],
        )

    def _human_declined(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        state.human_offer_declines += 1
        topic = state.human_offer_topic or "this"
        if state.human_offer_declines == 1:
            return ResponsePlan(
                action="reoffer_human",
                directive=(
                    f"{self._tone(ex)}Acknowledge their answer with understanding. Gently explain "
                    f"that speaking with a human representative is the recommended way to handle "
                    f"{topic}, because you aren't able to do it here. Do not pressure them."
                ),
                fallback_text=(
                    "I understand. Speaking with a human representative is the recommended way to "
                    "handle this, since I'm not able to do it here."
                ),
                closing_question=REOFFER_HUMAN_QUESTION,
                pending_question=Pending.OFFER_HUMAN,
                reasons=["caller declined the human offer once; one re-offer"],
            )

        # Second "no": respect it and move on.
        state.human_offer_topic = None
        state.human_offer_declines = 0
        if state.consent.status in ("denied", "no_response") and not state.verified:
            # A representative without consent has nothing else we can help with here.
            state.transition(Phase.ENDED)
            first = self._holder_first_name(state)
            return ResponsePlan(
                action="close",
                directive="",
                fallback_text=f"Okay. {first} is welcome to contact us directly anytime. Have a great day.",
                reasons=["representative without consent declined a human twice"],
                use_llm=False,
            )
        question, pending = self._after_declined_offer(state)
        return ResponsePlan(
            action="human_offer_declined",
            directive=f"{self._tone(ex)}Say okay, briefly and warmly. Do not mention the transfer again.",
            fallback_text="Okay.",
            closing_question=question,
            pending_question=pending,
            reasons=["caller declined the human offer twice; stop offering"],
        )

    def _after_declined_offer(self, state: SessionState) -> tuple[str, str | None]:
        if state.phase == Phase.POST_PROCESS:
            return self._email_question(state), Pending.OFFER_EMAIL
        if state.phase == Phase.VERIFY_ID:
            return self._identity_question(state), None
        if state.phase == Phase.RESOLVE_INTENT and self._claims(state) and not state.discussed:
            return WHICH_CLAIM_QUESTION, Pending.CHOOSE_CLAIM
        return ANYTHING_ELSE_QUESTION, Pending.ANYTHING_ELSE

    def _stuck_check(self, state: SessionState, ex: Extraction, plan: ResponsePlan) -> ResponsePlan:
        """Safety net: if we're about to ask the exact same question a 3rd time in a row,
        the conversation isn't progressing. Offer a human instead of looping."""
        repeatable = plan.pending_question in (Pending.OFFER_HUMAN, Pending.OFFER_EMAIL)
        if plan.closing_question and plan.closing_question == state.last_closing_question and not repeatable:
            state.repeat_count += 1
        else:
            state.repeat_count = 0
        state.last_closing_question = plan.closing_question

        if state.repeat_count < STUCK_REPEATS:
            return plan
        state.repeat_count = 0
        offer = self._offer_human(
            state, ex, topic="this",
            directive=(
                "Apologize briefly that you don't seem to be able to help with this here, and say "
                "a human representative can."
            ),
            fallback_text="I'm sorry, I don't seem to be able to help with this here, but a human representative can.",
            reason=f"stuck: same question asked {STUCK_REPEATS + 1} times ({plan.action})",
        )
        state.last_closing_question = offer.closing_question
        return offer

    def _escalate(self, state: SessionState, reason: str) -> ResponsePlan:
        state.escalate(reason)
        message = ESCALATION_MESSAGES[reason]
        return ResponsePlan(
            action="escalate",
            directive=(
                "You are transferring the caller to a human representative right now. Say so "
                f"warmly in 1-2 sentences, conveying this message: \"{message}\" "
                "Do not mention any limits, counts, or internal rules. Do not ask any question."
            ),
            fallback_text=message,
            reasons=[reason],
            use_llm=False,  # the last message must state exactly what happens; no paraphrase
        )

    def _resume(self, state: SessionState) -> tuple[str, str, str | None]:
        """How to steer back after a detour: (topic for the directive, question, pending)."""
        if state.pending_question == Pending.OFFER_HUMAN:
            return ("the question you asked.", REOFFER_HUMAN_QUESTION, Pending.OFFER_HUMAN)
        if state.phase == Phase.VERIFY_ID:
            return ("verifying their identity.", self._identity_question(state), None)
        if state.phase == Phase.RESOLVE_INTENT:
            # Re-ask whatever was open before the detour.
            if state.pending_question == Pending.CONFIRM_CLAIM and state.candidate_case_ids:
                claim = self.data.get_claim(state.verified_party_id, state.candidate_case_ids[0])
                return ("their claim.", self._confirm_question(claim), Pending.CONFIRM_CLAIM)
            return ("their claim.", WHICH_CLAIM_QUESTION, Pending.CHOOSE_CLAIM)
        if state.phase == Phase.PROCESS_CASE:
            return ("their claim.", ANYTHING_ELSE_QUESTION, Pending.ANYTHING_ELSE)
        return ("wrapping up.", self._email_question(state), Pending.OFFER_EMAIL)

    # --- Phase logic -------------------------------------------------------------------

    def _phase_plan(self, state: SessionState, ex: Extraction, user_text: str) -> ResponsePlan:
        if state.phase == Phase.VERIFY_ID:
            return self._verify(state, ex)
        if state.phase == Phase.RESOLVE_INTENT:
            return self._resolve(state, ex, user_text)
        if state.phase == Phase.PROCESS_CASE:
            return self._process(state, ex, user_text)
        return self._post_process(state, ex)

    # How a support agent responds to each emotion, before continuing with the workflow.
    # (Refusal is handled where it matters: VERIFY_ID offers the other identity fields.)
    _TONE = {
        Emotion.FRUSTRATED: (
            "The caller sounds frustrated. In one short sentence, acknowledge the situation (for "
            "example, that this has taken a few steps) without labeling their feelings or using "
            "stock phrases like 'I understand your frustration'. Then keep things moving. "
        ),
        Emotion.ANGRY: (
            "The caller sounds angry. Open with a brief, sincere apology for their experience, "
            "without blaming anyone and without naming their emotion. Stay calm and keep the "
            "reply short. "
        ),
        Emotion.ANXIOUS: (
            "The caller sounds worried. Reassure them briefly: their information is protected and "
            "you'll work through this with them. Then continue. "
        ),
        Emotion.CONFUSED: (
            "The caller sounds confused. Explain what you need or what is happening in simple, "
            "plain words, one thing at a time, and do not repeat your earlier wording. "
        ),
    }

    def _tone(self, ex: Extraction) -> str:
        return self._TONE.get(ex.emotion, "")

    # VERIFY_ID ----------------------------------------------------------------------------

    def _missing(self, state: SessionState) -> list[IdentityField]:
        return [f for f in IDENTITY_FIELDS if f not in state.identity]

    def _labels(self, fields: list[IdentityField]) -> list[str]:
        return [IDENTITY_FIELD_LABELS[f] for f in fields]

    def _identity_question(self, state: SessionState) -> str:
        labels = self._labels(self._missing(state))
        needed = max(REQUIRED_IDENTITY_MATCHES - len(state.identity), 1)
        if needed == 1:
            return f"Could you please share your {_join_or(labels)}?"
        return f"Could you please share {needed} more of the following: your {_join_or(labels)}?"

    def _verify(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        if state.consent.status == "pending":
            return self._consent_waiting(state, ex)
        if state.consent.status in ("denied", "no_response"):
            return self._consent_failed(state, ex)

        changed, unreadable = False, []
        for field in IDENTITY_FIELDS:
            raw = getattr(ex, field.value)
            if not raw:
                continue
            value = normalize_identity_value(field, raw)
            if value is None:
                unreadable.append(field)
            elif state.identity.get(field) != value:
                state.identity[field] = value
                changed = True

        result = verify_identity(state.identity, self.data.identity_records())
        reasons = [f"identity fields provided: {len(state.identity)}"]

        if result.verified:
            if state.caller_role == "representative":
                rep = self.data.find_representative(state.representative_name or "", result.party_id)
                if rep is None:
                    return self._escalate(state, "representative_not_authorized")
                return self._request_consent(state, result.party_id, rep)
            state.verified_party_id = result.party_id
            state.transition(Phase.RESOLVE_INTENT)
            return self._resolve(state, ex, "", just_verified=True)

        tone = self._tone(ex)
        explain_why = ""
        if (
            ex.emotion in _UPSET or ex.emotion == Emotion.ANXIOUS or ex.refuses_to_share
            or (ex.intent and not changed)
        ):
            explain_why = (
                "Explain briefly that claim details are protected and can only be shared after "
                "identity verification, to keep their information safe. Do not share any claim "
                "details. "
            )
        noted = ""
        if case_resolution.has_hints(state.hints):
            noted = (
                f"Let them know you've noted they're calling about their "
                f"{_describe_hints(state.hints)} and will look into it right after verification. "
            )
        unreadable_note = ""
        if unreadable:
            unreadable_note = (
                f"Say the {_join_or(self._labels(unreadable))} they gave could not be read (for an ID, "
                "only the last four digits are needed; for a date of birth, the full date). "
            )

        missing = self._missing(state)

        if result.evaluated and changed:
            state.counters.verification_failures += 1
            reasons.append(
                f"verification failed {state.counters.verification_failures}/{Limits.VERIFICATION_FAILURES}"
            )
            if state.counters.verification_failures >= Limits.VERIFICATION_FAILURES:
                return self._escalate(state, "verification_failed")
            if missing:
                return ResponsePlan(
                    action="verification_mismatch",
                    directive=(
                        f"{tone}Say you weren't able to verify their identity with the details so far, "
                        "and that one more detail would help. Do NOT say which detail did not match. "
                        f"{explain_why}"
                    ),
                    fallback_text="I wasn't able to verify your identity with the details provided so far.",
                    closing_question=f"To continue, could you also share your {_join_or(self._labels(missing))}?",
                    reasons=reasons,
                )
            return ResponsePlan(
                action="verification_mismatch",
                directive=(
                    f"{tone}Say you weren't able to verify their identity with those details. Do NOT "
                    "say which detail did not match."
                ),
                fallback_text="I'm sorry, I wasn't able to verify your identity with those details.",
                closing_question=(
                    "Could you please double-check them, using your full legal name and the "
                    "contact details on file?"
                ),
                reasons=reasons,
            )

        if ex.refuses_to_share:
            reasons.append("caller refused a field; offering alternatives")
        return ResponsePlan(
            action="ask_identity",
            directive=(
                f"{tone}{explain_why}{noted}{unreadable_note}"
                "Briefly say you need a few more details to verify their identity. If they refused "
                "one detail, reassure them that any of the others works instead."
            ),
            fallback_text=(
                f"{'I understand. ' if tone else 'Thank you. '}"
                f"{'Claim details are protected, so I need to verify your identity first.' if explain_why else 'I just need a bit more to verify your identity.'}"
            ),
            closing_question=self._identity_question(state),
            reasons=reasons,
        )

    # Representative consent ---------------------------------------------------------------
    #
    # A representative on file (e.g. David for Margaret) who gives 3 of the policyholder's
    # details still needs the policyholder's approval. The request goes to the
    # policyholder's phone; in the demo the tester answers it from the debug panel
    # (consent_decision). Nothing the caller types can approve it.

    def _holder_first_name(self, state: SessionState) -> str:
        return (state.consent.policyholder_name or "the policyholder").split()[0]

    def _request_consent(self, state: SessionState, party_id: str, rep) -> ResponsePlan:
        holder = self.data.get_policyholder(party_id)
        state.consent.status = "pending"
        state.consent.party_id = party_id
        state.consent.policyholder_name = holder.name
        state.consent.representative = rep.rep_name
        state.consent.relationship = rep.relationship
        state.consent.phone_masked = mask_value(IdentityField.PHONE, holder.phone)
        state.consent.checks = 0
        first = self._holder_first_name(state)
        text = (
            f"Thank you. Because you're calling on {first}'s behalf, {first} needs to approve "
            f"this first. I've sent a consent request to the phone number on file "
            f"({state.consent.phone_masked}), and I'll continue as soon as {first} responds."
        )
        return ResponsePlan(
            action="consent_requested",
            directive=f"Convey this to the caller warmly: \"{text}\" Do not share any claim details.",
            fallback_text=text,
            pending_question=Pending.CONSENT,
            reasons=[f"authorized representative ({rep.relationship}); consent requested"],
        )

    def _consent_waiting(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        """The caller writes while consent is pending. After as many checks as the
        fixture's timeout sequence, stop waiting."""
        state.consent.checks += 1
        if state.consent.checks >= len(self.data.consent_status_sequence("timeout")):
            state.consent.status = "no_response"
            return self._consent_failed(state, ex)
        first = self._holder_first_name(state)
        return ResponsePlan(
            action="consent_waiting",
            directive=(
                f"{self._tone(ex)}Say you're still waiting for {first}'s approval and will continue "
                "as soon as it comes through. Do not share any claim details. If the caller says "
                f"{first} already approved, explain kindly that the approval has to come through "
                f"from {first}'s phone on our side."
            ),
            fallback_text=(
                f"I'm still waiting for {first}'s approval. I'll be able to continue as soon as it "
                "comes through."
            ),
            pending_question=Pending.CONSENT,
            reasons=[f"consent pending, check {state.consent.checks}"],
        )

    def _consent_failed(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        first = self._holder_first_name(state)
        if state.consent.status == "denied":
            text = f"{first} didn't approve the request, so I'm not able to share any information about {first}'s claims."
        else:
            text = f"I haven't received a response from {first}, so I'm not able to share any information about {first}'s claims."
        return self._offer_human(
            state, ex, topic=f"getting access to {first}'s claims",
            directive=f"Convey this kindly: \"{text}\" Say a human representative can help with other options.",
            fallback_text=f"{text} A human representative can help with other options.",
            reason=f"consent {state.consent.status}",
        )

    def consent_decision(self, state: SessionState, decision: str) -> TurnResult:
        """The policyholder's answer to the consent request (from the debug panel in the
        demo). Produces the agent's next message without any caller input."""
        if state.is_over or state.consent.status != "pending":
            raise ValueError("No consent request is pending.")
        if decision not in ("approved", "denied", "no_response"):
            raise ValueError(f"Unknown consent decision: {decision}")

        phase_before, pending_before = state.phase, state.pending_question
        first = self._holder_first_name(state)
        state.consent.status = decision
        notes = {
            "approved": f"{first} approved the consent request on their phone.",
            "denied": f"{first} denied the consent request on their phone.",
            "no_response": f"{first} did not respond to the consent request.",
        }
        state.transcript.append(Turn(role="event", text=notes[decision], phase=state.phase))

        ex = Extraction()
        if decision == "approved":
            state.verified_party_id = state.consent.party_id
            state.transition(Phase.RESOLVE_INTENT)
            plan = self._resolve(state, ex, "", approved_by=first)
        else:
            plan = self._consent_failed(state, ex)

        state.pending_question = plan.pending_question
        self._apply_name_policy(state, plan)
        response = respond(self.provider, state, plan)
        state.transcript.append(Turn(role="agent", text=response.text, phase=state.phase))

        trace = self._trace(
            state, phase_before, pending_before, f"[policyholder consent: {decision}]", None, plan, response
        )
        self.tracer.write(state.session_id, trace)
        return TurnResult(response.text, trace)

    # RESOLVE_INTENT -----------------------------------------------------------------------

    def _claims(self, state: SessionState) -> list[Claim]:
        return self.data.list_claims(state.verified_party_id)

    def _confirm_question(self, claim: Claim) -> str:
        return f"Just to confirm, are you calling about {case_resolution.describe_claim(claim)}?"

    def _select(self, state: SessionState, claim: Claim, ex: Extraction, user_text: str) -> ResponsePlan:
        state.selected_case_id = claim.case_id
        state.candidate_case_ids = []
        state.transition(Phase.PROCESS_CASE)
        return self._process(state, ex, user_text, just_selected=True)

    def _confirm(
        self, state: SessionState, claim: Claim, ex: Extraction, opener: str, opener_text: str,
        reasons: list[str],
    ) -> ResponsePlan:
        """Every claim inferred from a description is confirmed before it's discussed."""
        state.candidate_case_ids = [claim.case_id]
        return ResponsePlan(
            action="confirm_claim",
            directive=(
                f"{self._tone(ex)}{opener}Say you'd like to make sure you have the right claim. "
                "Do not share any claim details yet."
            ),
            fallback_text=opener_text or "Thank you.",
            facts=[f"Candidate: {case_resolution.describe_claim(claim)}."],
            closing_question=self._confirm_question(claim),
            pending_question=Pending.CONFIRM_CLAIM,
            reasons=reasons,
        )

    def _resolve(
        self, state: SessionState, ex: Extraction, user_text: str, just_verified: bool = False,
        approved_by: str | None = None,
    ) -> ResponsePlan:
        claims = self._claims(state)
        by_id = {c.case_id: c for c in claims}
        opener = "Thank them; their identity is now verified. " if just_verified else ""
        opener_text = "Thank you, you're verified." if just_verified else ""
        if approved_by:
            opener = (
                f"Thank them for waiting and tell them {approved_by} has approved the request, so "
                f"you can now help with {approved_by}'s claims. "
            )
            opener_text = (
                f"Thank you for waiting. {approved_by} has approved the request, so I can help "
                f"with {approved_by}'s claims."
            )
        turn_hints = _hints_from(ex)

        if not claims:
            return self._no_claims(state, ex, opener, opener_text, just_verified)

        if state.pending_question == Pending.CONFIRM_CLAIM and state.candidate_case_ids:
            candidate = by_id[state.candidate_case_ids[0]]
            # "Yes that one" may come with hints restating the claim; only a contradiction
            # (e.g. "yes, the dental one" when we asked about healthcare) blocks the yes.
            consistent = bool(case_resolution.match_claims([candidate], turn_hints)) or not (
                case_resolution.has_hints(turn_hints)
            )
            if _answer(ex) == YesNo.YES and consistent:
                return self._select(state, candidate, ex, user_text)
            if _answer(ex) == YesNo.NO or not consistent:
                state.candidate_case_ids = []
                state.hints = turn_hints  # forget the wrong guess, keep only what was just said

        # The caller named an exact claim ID: no need to confirm.
        if turn_hints.case_id and turn_hints.case_id in by_id:
            return self._select(state, by_id[turn_hints.case_id], ex, user_text)

        # Choosing among a list we offered: match this turn's words against those claims only.
        if state.pending_question == Pending.CHOOSE_CLAIM and case_resolution.has_hints(turn_hints):
            offered = [by_id[i] for i in state.candidate_case_ids if i in by_id]
            picked = case_resolution.match_claims(offered, turn_hints)
            if len(picked) == 1:
                return self._confirm(
                    state, picked[0], ex, opener, opener_text, ["picked from offered list"]
                )

        if not case_resolution.has_hints(state.hints):
            return self._ask_which_claim(state, claims, ex, opener, opener_text)

        matches = case_resolution.match_claims(claims, state.hints)
        if len(matches) == 1:
            remembered = (
                f"Mention you're following up on the {_describe_hints(state.hints)} they mentioned earlier. "
                if just_verified else ""
            )
            return self._confirm(
                state, matches[0], ex, opener + remembered, opener_text,
                ["one claim matches remembered hints", _describe_hints(state.hints)],
            )
        if len(matches) > 1:
            return self._narrow(state, matches, ex, opener, opener_text)

        # Nothing matches what they described. Say so, without revealing what they do have.
        described = _describe_hints(state.hints)
        state.hints = CaseHints()
        state.candidate_case_ids = [c.case_id for c in claims]
        return ResponsePlan(
            action="no_matching_claim",
            directive=(
                f"{self._tone(ex)}{opener}Say you couldn't find a {described} on their account. "
                "Do not list or describe any of their claims."
            ),
            fallback_text=f"{opener_text} I couldn't find a {described} on your account.".strip(),
            closing_question=(
                "Could you tell me a bit more about the claim, such as its claim number, what type "
                "of claim it is, when it was filed, or its status?"
            ),
            pending_question=Pending.CHOOSE_CLAIM,
            reasons=[f"no claim matches: {described}"],
        )

    # Claims are never listed: the caller describes the claim, the code finds it, and
    # only the one claim they pointed to is ever named (in the confirmation question).

    def _ask_which_claim(
        self, state: SessionState, claims: list[Claim], ex: Extraction, opener: str, opener_text: str
    ) -> ResponsePlan:
        state.candidate_case_ids = [c.case_id for c in claims]
        owner = (
            f"{self._holder_first_name(state)} has" if state.consent.status == "approved" else "you have"
        )
        return ResponsePlan(
            action="ask_which_claim",
            directive=(
                f"{self._tone(ex)}{opener}Say you can see there are some claims with us. Do NOT "
                "list, count, or describe any of the claims."
            ),
            fallback_text=f"{opener_text} I see {owner} some claims with us.".strip(),
            closing_question=WHICH_CLAIM_QUESTION,
            pending_question=Pending.CHOOSE_CLAIM,
            reasons=[f"{len(claims)} claims on file; caller hasn't said which"],
        )

    def _narrow(
        self, state: SessionState, matches: list[Claim], ex: Extraction, opener: str, opener_text: str
    ) -> ResponsePlan:
        """Several claims fit the description: ask for a detail that separates them."""
        state.candidate_case_ids = [c.case_id for c in matches]
        described = _describe_hints(state.hints)
        details = case_resolution.distinguishing_details(matches)
        return ResponsePlan(
            action="narrow_claim",
            directive=(
                f"{self._tone(ex)}{opener}Say you see more than one {described} on their account. "
                "Do NOT list or describe the claims."
            ),
            fallback_text=f"{opener_text} I see more than one {described} on your account.".strip(),
            closing_question=f"To find the right one, could you tell me {_join_or(details)}?",
            pending_question=Pending.CHOOSE_CLAIM,
            reasons=[f"{len(matches)} claims match: {described}", f"ask for: {details}"],
        )

    def _no_claims(
        self, state: SessionState, ex: Extraction, opener: str, opener_text: str,
        just_verified: bool,
    ) -> ResponsePlan:
        """A verified caller with nothing on file. Ask how we can help; a claim question or
        an unsupported request (handled by the guards) leads to a human offer."""
        done = ex.wants_to_end or (
            state.pending_question == Pending.ANYTHING_ELSE and _answer(ex) == YesNo.NO
        )
        if done:
            state.transition(Phase.ENDED)
            return ResponsePlan(
                action="close",
                directive=(
                    "Say that's no problem, they're welcome to contact us anytime, and say "
                    "goodbye. Do not ask any question."
                ),
                fallback_text="No problem. You're welcome to contact us anytime. Have a great day.",
                reasons=["no claims on file; caller is done"],
                use_llm=False,
            )

        asked_about_claim = case_resolution.has_hints(_hints_from(ex)) or bool(
            ex.intent and ex.intent not in (Intent.UNKNOWN, Intent.SPEAK_TO_HUMAN)
        )
        if asked_about_claim and not just_verified:
            return self._offer_human(
                state, ex, topic="a claim that isn't on file",
                directive=(
                    "Say you don't see any claims on file for their account, so there's nothing for "
                    "you to look up. Do not suggest that any claim exists. Say a human "
                    "representative can help, for example if a claim hasn't been filed yet."
                ),
                fallback_text=(
                    "I don't see any claims on file for your account, so there's nothing for me to "
                    "look up. A human representative can help, for example if a claim hasn't been "
                    "filed yet."
                ),
                facts=["The caller's account has no claims on file."],
                reason="caller asked about a claim, but none are on file",
            )

        return ResponsePlan(
            action="no_claims_on_file",
            directive=(
                f"{self._tone(ex)}{'Thank them for verifying their identity. ' if just_verified else ''}"
                "Say you don't see any existing claims with us. Do not suggest that any claim exists."
            ),
            fallback_text=(
                f"{'Thank you for verifying your identity. ' if just_verified else ''}"
                "I don't see any existing claims with us."
            ),
            facts=["The caller's account has no claims on file."],
            closing_question="What can I help you with today?",
            pending_question=Pending.ANYTHING_ELSE,
            reasons=["verified caller has no claims on file"],
        )

    # PROCESS_CASE -------------------------------------------------------------------------

    def _process(
        self, state: SessionState, ex: Extraction, user_text: str, just_selected: bool = False
    ) -> ResponsePlan:
        claim = self.data.get_claim(state.verified_party_id, state.selected_case_id)
        new_question = bool(ex.intent and ex.intent != Intent.UNKNOWN) or bool(ex.followup_topic)

        if not just_selected:
            done = ex.wants_to_end or (
                state.pending_question == Pending.ANYTHING_ELSE and _answer(ex) == YesNo.NO
            )
            if done and not new_question:
                state.transition(Phase.POST_PROCESS)
                return self._offer_email(state, ex)

            turn_hints = _hints_from(ex)
            switching = (turn_hints.case_id and turn_hints.case_id != claim.case_id) or (
                turn_hints.case_type and turn_hints.case_type != claim.case_type
            )
            if switching:
                state.selected_case_id = None
                state.hints = turn_hints
                state.transition(Phase.RESOLVE_INTENT)
                return self._resolve(state, ex, user_text)

        intent = ex.intent if ex.intent and ex.intent != Intent.UNKNOWN else state.intent
        intent = intent or Intent.STATUS_INQUIRY
        question = "" if just_selected else user_text
        answer = grounding.build_answer(
            self.data, claim, intent, question, ex.followup_topic, self.today()
        )
        state.discussed.append(
            DiscussedItem(case_id=claim.case_id, intent=intent, facts=answer.facts, next_steps=answer.next_steps)
        )
        facts = answer.facts + [f"Next step: {s}" for s in answer.next_steps]

        human = ""
        closing, pending = ANYTHING_ELSE_QUESTION, Pending.ANYTHING_ELSE
        if answer.needs_human:
            human = (
                "Explain that, according to the records, the appeal deadline has passed, so a "
                "human claims representative needs to review their options. "
            )
            closing, pending = DEADLINE_HUMAN_QUESTION, Pending.OFFER_HUMAN
            state.human_offer_topic = "a claim whose appeal deadline has passed"
            state.human_offer_for_request = False
            state.human_offer_declines = 0

        return ResponsePlan(
            action="answer_case",
            directive=(
                f"{self._tone(ex)}{'Thank them for confirming. ' if just_selected else ''}"
                f"The caller's question is about: {intent.value.replace('_', ' ')}. Answer it "
                f"naturally using ONLY the FACTS. {human}"
            ),
            fallback_text=" ".join(facts),
            facts=facts,
            closing_question=closing,
            pending_question=pending,
            reasons=[f"intent={intent.value}", f"topics={answer.topics_used}", f"needs_human={answer.needs_human}"],
        )

    # POST_PROCESS -------------------------------------------------------------------------

    def _masked_email(self, state: SessionState) -> str:
        email = self.data.get_policyholder(state.verified_party_id).email
        return mask_value(IdentityField.EMAIL, email)

    def _email_question(self, state: SessionState) -> str:
        return (
            "Would you like me to email you a summary of our conversation, including your claim "
            f"status and next steps, to {self._masked_email(state)}?"
        )

    def _wrap_up_facts(self, state: SessionState) -> list[str]:
        """What was established in this call, so wrap-up questions can still be answered."""
        if not state.discussed:
            return []
        last = state.discussed[-1]
        return last.facts + [f"Next step: {s}" for s in last.next_steps]

    def _offer_email(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        state.email.offered = True
        handoff = (
            "Tell them you'll connect them with a human representative right after this. "
            if state.handoff_requested else ""
        )
        handoff_text = (
            "I'll connect you with a human representative right after this. "
            if state.handoff_requested else ""
        )
        return ResponsePlan(
            action="offer_email",
            directive=(
                f"{self._tone(ex)}{handoff}Say you can email them a summary of this conversation "
                "(what was discussed, the claim status, and next steps), and that they can see it "
                "first with the 'Preview email' button."
            ),
            fallback_text=f"{handoff_text}You can check the summary first with the 'Preview email' button.",
            facts=self._wrap_up_facts(state),
            closing_question=self._email_question(state),
            pending_question=Pending.OFFER_EMAIL,
            reasons=["case handled; offering email summary", f"handoff={state.handoff_requested}"],
        )

    def _post_process(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        answer = _answer(ex) if state.pending_question == Pending.OFFER_EMAIL else None
        if answer is None:
            return ResponsePlan(
                action="ask_email_again",
                directive=(
                    f"{self._tone(ex)}If the caller asked a question, answer it briefly using ONLY "
                    "the FACTS (for example, why a human representative is needed). Otherwise just "
                    "say you'd like to wrap up."
                ),
                fallback_text="Before we wrap up:",
                facts=self._wrap_up_facts(state),
                closing_question=self._email_question(state),
                pending_question=Pending.OFFER_EMAIL,
                reasons=["email consent not answered yet"],
            )

        state.email.consent = answer == YesNo.YES
        sent = (
            f"I've sent the summary to {self._masked_email(state)}. "
            if state.email.consent else "No problem, I won't send an email. "
        )
        if state.handoff_requested:
            state.escalate("caller_requested_human")
            goodbye = "I'm connecting you with a human representative now. Please hold for a moment."
        else:
            state.transition(Phase.ENDED)
            goodbye = "Thank you for calling, and have a great day."
        return ResponsePlan(
            action="close",
            directive=(
                f"Confirm this to the caller and close the conversation: \"{sent}{goodbye}\" "
                "Do not ask any question."
            ),
            fallback_text=sent + goodbye,
            reasons=[f"email consent={state.email.consent}", f"handoff={state.handoff_requested}"],
            use_llm=False,  # states exactly where the email went and what happens next
        )

    # --- Trace -------------------------------------------------------------------------

    def _trace(self, state, phase_before, pending_before, user_text, outcome, plan, response) -> dict:
        record = {
            "turn": sum(1 for t in state.transcript if t.role == "user"),
            "phase_before": phase_before.value,
            "phase_after": state.phase.value,
            "pending_question_before": pending_before,
            "user_text": mask_text(user_text),
            "plan": {
                "action": plan.action,
                "reasons": plan.reasons,
                "directive": mask_text(plan.directive),
                "facts": plan.facts,
                "closing_question": mask_text(plan.closing_question or ""),
                "pending_question": plan.pending_question,
            },
            "response": {
                "source": response.source,
                "guard_violations": response.guard_violations,
                "retried": response.retried,
                "llm_latency_ms": response.llm.latency_ms if response.llm else None,
                "llm_error": response.llm.error if response.llm else None,
            },
            "reply": mask_text(response.text),
            "state": public_snapshot(state),
        }
        if outcome is not None:
            record["extraction"] = {
                "parse_layer": outcome.parse_layer,
                "regex_fields": mask_fields(outcome.regex_fields),
                "fields": mask_fields(outcome.extraction.model_dump(exclude_defaults=True, mode="json")),
                "llm_latency_ms": outcome.llm.latency_ms if outcome.llm else None,
                "llm_error": outcome.llm.error if outcome.llm else None,
                "raw_output": mask_text(outcome.llm.text[:2000]) if outcome.llm else None,
            }
        return record

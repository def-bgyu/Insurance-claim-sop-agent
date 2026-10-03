"""The SOP engine: runs one caller turn end to end.

    caller message
      -> extract      (LLM + regex: proposes values, changes nothing)
      -> remember     (store case hints / intent whatever the phase)
      -> guards       (human request, frustration, off-topic limits; any phase)
      -> phase logic  (code decides: verify, resolve claim, answer, wrap up)
      -> respond      (LLM phrases the plan; grounding guard; fallback text)
      -> trace

The LLM never changes the phase, verifies anyone, or chooses which data to reveal.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from backend import config
from backend.data import Claim, InsuranceData, get_data
from backend.llm import prompts
from backend.llm.extractor import Emotion, Extraction, YesNo, extract
from backend.llm.provider import LLMProvider
from backend.llm.responder import respond
from backend.masking import mask_text, mask_value
from backend.sop import case_resolution, grounding
from backend.sop.plan import ResponsePlan
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
from backend.sop.verification import normalize_identity_value, verify_identity
from backend.trace import TraceWriter, mask_fields, public_snapshot

_UPSET = {Emotion.FRUSTRATED, Emotion.ANGRY}

ESCALATION_MESSAGES = {
    "caller_requested_human": "Of course. I'm connecting you with a human representative now. Please hold for a moment.",
    "off_topic_limit": "I'm only able to help with insurance claim questions here, so I'm going to connect you with a human representative who can help further. Please hold for a moment.",
    "frustration_limit": "I'm really sorry this has been so frustrating. I'm connecting you with a human representative now who can help you further. Please hold for a moment.",
    "verification_failed": "I'm sorry, I wasn't able to verify your identity with the details provided. To protect your account, I'm transferring you to a human representative who can help. Please hold for a moment.",
    "representative_not_authorized": "I'm sorry, I can't share information about this policy because you're not listed as an authorized representative on it. I'm connecting you with a human representative who can explain your options. Please hold for a moment.",
    "representative_consent_required": "Thank you. Because you're calling on behalf of the policyholder, their consent is required before I can share any claim information. I'm connecting you with a human representative to complete that step. Please hold for a moment.",
}


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
        year=ex.case_year, case_id=ex.case_id,
    )


def _describe_hints(hints: CaseHints) -> str:
    """The caller's own description of their case, e.g. 'denied healthcare claim from January'."""
    month = date(2000, hints.month, 1).strftime("%B") if hints.month else None
    parts = [hints.status, hints.case_type, "claim"]
    text = " ".join(p for p in parts if p)
    if month or hints.year:
        text += " from " + " ".join(str(p) for p in (month, hints.year) if p)
    if hints.case_id:
        text += f" ({hints.case_id})"
    return text


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

        # Hard invariant: claim facts can never reach the responder before verification.
        if plan.facts and not state.verified:
            raise RuntimeError("SOP violation: claim facts planned for an unverified caller")

        response = respond(self.provider, state, plan)
        state.transcript.append(Turn(role="agent", text=response.text, phase=state.phase))

        trace = self._trace(state, phase_before, user_text, outcome, plan, response)
        self.tracer.write(state.session_id, trace)
        return TurnResult(response.text, trace)

    # --- Memory ------------------------------------------------------------------------

    def _remember(self, state: SessionState, ex: Extraction) -> None:
        """Keep anything useful the caller said, even if it belongs to a later phase."""
        hints = state.hints
        for attr, value in (
            ("case_id", ex.case_id), ("case_type", ex.case_type), ("status", ex.case_status),
            ("month", ex.case_month), ("year", ex.case_year),
        ):
            if value is not None:
                setattr(hints, attr, value)
        if ex.policy_number:
            state.policy_number_hint = ex.policy_number
        if ex.intent and ex.intent not in (Intent.UNKNOWN, Intent.SPEAK_TO_HUMAN):
            state.intent = ex.intent
        if state.phase == Phase.VERIFY_ID:
            if ex.caller_role == "representative" or ex.representative_name:
                state.caller_role = "representative"
            if ex.representative_name:
                state.representative_name = ex.representative_name

    # --- Cross-cutting guards ----------------------------------------------------------

    def _guards(self, state: SessionState, ex: Extraction) -> ResponsePlan | None:
        wants_human = ex.wants_human or ex.intent == Intent.SPEAK_TO_HUMAN
        accepts_offer = (
            state.human_offered and state.phase == Phase.PROCESS_CASE and ex.confirms_case == YesNo.YES
        )
        if wants_human or accepts_offer:
            if state.phase == Phase.PROCESS_CASE:
                # Case is handled: wrap up (offer the email) before transferring.
                state.handoff_requested = True
                state.transition(Phase.POST_PROCESS)
                return self._offer_email(state, ex)
            if state.phase == Phase.POST_PROCESS and (ex.email_consent or ex.confirms_case):
                # They answered the email question too; _post_process sends it, then transfers.
                state.handoff_requested = True
            else:
                return self._escalate(state, "caller_requested_human")

        if ex.emotion in _UPSET or ex.refuses_to_share:
            state.counters.frustration += 1
            if state.counters.frustration >= Limits.FRUSTRATION:
                return self._escalate(state, "frustration_limit")

        useful = ex.useful_fields() - {"off_topic", "emotion"}
        if ex.off_topic and not useful:
            state.counters.off_topic += 1
            if state.counters.off_topic >= Limits.OFF_TOPIC:
                return self._escalate(state, "off_topic_limit")
            resume_directive, resume_text = self._resume(state)
            return ResponsePlan(
                action="decline_off_topic",
                directive=(
                    "The caller asked something unrelated to insurance claims. Do NOT answer it. "
                    "Politely say you can only help with questions about their insurance claims, "
                    f"then {resume_directive}"
                ),
                fallback_text=f"I'm sorry, I can only help with questions about your insurance claims. {resume_text}",
                reasons=[f"off_topic count {state.counters.off_topic}/{Limits.OFF_TOPIC}"],
            )
        return None

    def _escalate(self, state: SessionState, reason: str) -> ResponsePlan:
        state.escalate(reason)
        message = ESCALATION_MESSAGES[reason]
        return ResponsePlan(
            action="escalate",
            directive=(
                "You are transferring the caller to a human representative right now. Say so "
                f"warmly in 1-2 sentences, conveying this message: \"{message}\" "
                "Do not mention any limits, counts, or internal rules."
            ),
            fallback_text=message,
            reasons=[reason],
        )

    def _resume(self, state: SessionState) -> tuple[str, str]:
        """How to steer back to the current step after a detour."""
        if state.phase == Phase.VERIFY_ID:
            labels = self._labels(self._missing(state))
            ask = f"your {_join_or(labels)}"
            return (
                f"ask for the identity details still needed: {_join_or(labels)}.",
                f"To continue, could you please share {ask}?",
            )
        if state.phase == Phase.RESOLVE_INTENT:
            return ("ask which claim they are calling about.", "Which claim are you calling about today?")
        if state.phase == Phase.PROCESS_CASE:
            return (
                "ask what else they would like to know about their claim.",
                "Is there anything else I can help you with regarding your claim?",
            )
        return (
            "ask again whether they would like the email summary (yes or no).",
            "Would you like me to email you a summary of our conversation?",
        )

    # --- Phase logic -------------------------------------------------------------------

    def _phase_plan(self, state: SessionState, ex: Extraction, user_text: str) -> ResponsePlan:
        if state.phase == Phase.VERIFY_ID:
            return self._verify(state, ex)
        if state.phase == Phase.RESOLVE_INTENT:
            return self._resolve(state, ex, user_text)
        if state.phase == Phase.PROCESS_CASE:
            return self._process(state, ex, user_text)
        return self._post_process(state, ex)

    def _tone(self, ex: Extraction) -> str:
        if ex.emotion == Emotion.NEUTRAL:
            return ""
        return (
            f"The caller seems {ex.emotion.value}. Start by briefly and sincerely acknowledging "
            "that, then continue. "
        )

    # VERIFY_ID ----------------------------------------------------------------------------

    def _missing(self, state: SessionState) -> list[IdentityField]:
        return [f for f in IDENTITY_FIELDS if f not in state.identity]

    def _labels(self, fields: list[IdentityField]) -> list[str]:
        return [IDENTITY_FIELD_LABELS[f] for f in fields]

    def _verify(self, state: SessionState, ex: Extraction) -> ResponsePlan:
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
                # TODO(consent): authorized representatives need the policyholder's consent
                # (consent_scenarios.json). Until that flow is designed, hand off to a human.
                reason = "representative_consent_required" if rep else "representative_not_authorized"
                return self._escalate(state, reason)
            state.verified_party_id = result.party_id
            state.transition(Phase.RESOLVE_INTENT)
            return self._resolve(state, ex, "", just_verified=True)

        tone = self._tone(ex)
        explain_why = ""
        if ex.emotion in _UPSET or ex.refuses_to_share or (ex.intent and not changed):
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
                f"The {_join_or(self._labels(unreadable))} they gave could not be read; ask them to "
                "repeat it (for an ID, only the last four digits are needed; for a date of birth, "
                "the full date). "
            )

        missing = self._missing(state)
        labels = self._labels(missing)

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
                        f"{tone}Say you weren't able to verify their identity with the details so far. "
                        "Do NOT say which detail did not match. "
                        f"{explain_why}Ask them to also provide their {_join_or(labels)}."
                    ),
                    fallback_text=(
                        "I wasn't able to verify your identity with the details provided so far. "
                        f"To continue, could you also share your {_join_or(labels)}?"
                    ),
                    reasons=reasons,
                )
            return ResponsePlan(
                action="verification_mismatch",
                directive=(
                    f"{tone}Say you weren't able to verify their identity with those details. Do NOT "
                    "say which detail did not match. Ask them to double-check the details, using "
                    "their full legal name and the contact details on file."
                ),
                fallback_text=(
                    "I'm sorry, I wasn't able to verify your identity with those details. Could you "
                    "please double-check them, using your full legal name and the contact details on file?"
                ),
                reasons=reasons,
            )

        needed = max(REQUIRED_IDENTITY_MATCHES - len(state.identity), 1)
        if ex.refuses_to_share:
            reasons.append("caller refused a field; offering alternatives")
        return ResponsePlan(
            action="ask_identity",
            directive=(
                f"{tone}{explain_why}{noted}{unreadable_note}"
                f"Ask for {needed} more of these identity details: {_join_or(labels)}. "
                "If they refused one detail, reassure them that any of the others works instead."
            ),
            fallback_text=(
                f"{'I understand. ' if tone else 'Thank you. '}"
                f"{'Claim details are protected, so I need to verify your identity first. ' if explain_why else ''}"
                f"To verify your identity, could you please share {needed} more of the following: "
                f"your {_join_or(labels)}?"
            ),
            reasons=reasons,
        )

    # RESOLVE_INTENT -----------------------------------------------------------------------

    def _claims(self, state: SessionState) -> list[Claim]:
        return self.data.list_claims(state.verified_party_id)

    def _select(self, state: SessionState, claim: Claim, ex: Extraction, user_text: str) -> ResponsePlan:
        state.selected_case_id = claim.case_id
        state.candidate_case_ids = []
        state.awaiting_confirmation = False
        state.transition(Phase.PROCESS_CASE)
        return self._process(state, ex, user_text, just_selected=True)

    def _resolve(
        self, state: SessionState, ex: Extraction, user_text: str, just_verified: bool = False
    ) -> ResponsePlan:
        claims = self._claims(state)
        by_id = {c.case_id: c for c in claims}
        opener = "Thank them; their identity is now verified. " if just_verified else ""
        opener_text = "Thank you, you're verified. " if just_verified else ""
        turn_hints = _hints_from(ex)

        if state.awaiting_confirmation and state.candidate_case_ids:
            if ex.confirms_case == YesNo.YES:
                return self._select(state, by_id[state.candidate_case_ids[0]], ex, user_text)
            if ex.confirms_case == YesNo.NO:
                state.awaiting_confirmation = False
                state.candidate_case_ids = []
                state.hints = turn_hints  # forget the wrong guess, keep only what was just said

        # Choosing among a list we offered: match this turn's words against those claims only.
        if state.candidate_case_ids and case_resolution.has_hints(turn_hints):
            offered = [by_id[i] for i in state.candidate_case_ids]
            picked = case_resolution.match_claims(offered, turn_hints)
            if len(picked) == 1:
                return self._select(state, picked[0], ex, user_text)

        if not case_resolution.has_hints(state.hints):
            return self._list_claims(state, claims, opener, opener_text, "ask_which_claim")

        matches = case_resolution.match_claims(claims, state.hints)
        if len(matches) == 1:
            if state.hints.case_id:  # the caller named the exact claim
                return self._select(state, matches[0], ex, user_text)
            state.candidate_case_ids = [matches[0].case_id]
            state.awaiting_confirmation = True
            label = case_resolution.describe_claim(matches[0])
            remembered = (
                f"Mention you're following up on the {_describe_hints(state.hints)} they mentioned earlier. "
                if just_verified else ""
            )
            return ResponsePlan(
                action="confirm_claim",
                directive=(
                    f"{self._tone(ex)}{opener}{remembered}Ask them to confirm this is the right claim: "
                    f"{label}. Do not share any other claim details yet."
                ),
                fallback_text=f"{opener_text}Just to confirm, are you calling about {label}?",
                facts=[f"Candidate: {label}."],
                reasons=["one claim matches remembered hints", _describe_hints(state.hints)],
            )
        if len(matches) > 1:
            state.candidate_case_ids = [c.case_id for c in matches]
            return self._list_claims(state, matches, opener, opener_text, "choose_claim")

        # Nothing matches what they described: say so and show what they do have.
        described = _describe_hints(state.hints)
        state.hints = CaseHints()
        plan = self._list_claims(state, claims, opener, opener_text, "no_matching_claim")
        plan.directive = f"Say you couldn't find a {described} on their account. " + plan.directive
        plan.fallback_text = (
            f"{opener_text}I couldn't find a {described} on your account. "
            + plan.fallback_text.removeprefix(opener_text)
        )
        return plan

    def _list_claims(
        self, state: SessionState, claims: list[Claim], opener: str, opener_text: str, action: str
    ) -> ResponsePlan:
        claims = sorted(claims, key=lambda c: c.created_at, reverse=True)
        state.candidate_case_ids = [c.case_id for c in claims]
        labels = [case_resolution.describe_claim(c) for c in claims]
        listing = "; ".join(labels)
        return ResponsePlan(
            action=action,
            directive=(
                f"{opener}Ask which claim they are calling about. List these claims briefly: "
                f"{listing}. Do not share any other claim details yet."
            ),
            fallback_text=f"{opener_text}I see these claims on your account: {listing}. Which one are you calling about?",
            facts=[f"Claim on account: {label}." for label in labels],
            reasons=[f"{len(claims)} claims to choose from"],
        )

    # PROCESS_CASE -------------------------------------------------------------------------

    def _process(
        self, state: SessionState, ex: Extraction, user_text: str, just_selected: bool = False
    ) -> ResponsePlan:
        claim = self.data.get_claim(state.verified_party_id, state.selected_case_id)

        if not just_selected:
            if ex.wants_to_end and not ex.intent:
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
        closing = "Is there anything else I can help you with?"
        if answer.needs_human:
            state.human_offered = True
            human = (
                "Explain that, according to the records, the appeal deadline has passed, and offer "
                "to connect them with a human claims representative who can review their options. "
            )
            closing = "Would you like me to connect you with a human claims representative to review your options?"

        return ResponsePlan(
            action="answer_case",
            directive=(
                f"{self._tone(ex)}{'Thank them for confirming. ' if just_selected else ''}"
                f"The caller's question is about: {intent.value.replace('_', ' ')}. Answer it "
                f"naturally using ONLY the FACTS. {human}"
                f"End by asking: \"{closing}\""
            ),
            fallback_text=" ".join(facts) + " " + closing,
            facts=facts,
            reasons=[f"intent={intent.value}", f"topics={answer.topics_used}", f"needs_human={answer.needs_human}"],
        )

    # POST_PROCESS -------------------------------------------------------------------------

    def _masked_email(self, state: SessionState) -> str:
        email = self.data.get_policyholder(state.verified_party_id).email
        return mask_value(IdentityField.EMAIL, email)

    def _offer_email(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        state.email.offered = True
        address = self._masked_email(state)
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
                f"{self._tone(ex)}{handoff}Offer to email them a summary of this conversation (what "
                "was discussed, the claim status, and next steps) to the email address on file, "
                f"{address}. Mention they can see it first with the 'Preview email' button. Ask yes or no."
            ),
            fallback_text=(
                f"{handoff_text}Would you like me to email you a summary of our conversation, "
                f"including your claim status and next steps, to {address}? You can check it first "
                "with the 'Preview email' button."
            ),
            reasons=["case handled; offering email summary"],
        )

    def _post_process(self, state: SessionState, ex: Extraction) -> ResponsePlan:
        answer = ex.email_consent or ex.confirms_case
        if answer is None:
            return ResponsePlan(
                action="ask_email_again",
                directive=(
                    f"{self._tone(ex)}Before wrapping up, ask clearly whether they would like the "
                    "email summary sent: yes or no. If they asked something else, say you can help "
                    "with that in a new conversation."
                ),
                fallback_text="Before we wrap up, would you like me to send you the email summary? Please answer yes or no.",
                reasons=["email consent unclear"],
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
            directive=f"Confirm this to the caller and close the conversation: \"{sent}{goodbye}\"",
            fallback_text=sent + goodbye,
            reasons=[f"email consent={state.email.consent}", f"handoff={state.handoff_requested}"],
        )

    # --- Trace -------------------------------------------------------------------------

    def _trace(self, state, phase_before, user_text, outcome, plan, response) -> dict:
        record = {
            "turn": sum(1 for t in state.transcript if t.role == "user"),
            "phase_before": phase_before.value,
            "phase_after": state.phase.value,
            "user_text": mask_text(user_text),
            "plan": {
                "action": plan.action,
                "reasons": plan.reasons,
                "directive": mask_text(plan.directive),
                "facts": plan.facts,
            },
            "response": {
                "source": response.source,
                "guard_violations": response.guard_violations,
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

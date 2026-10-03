"""End-to-end SOP scenarios with a scripted model. These check the harness, not the LLM."""

from datetime import date

from backend.sop.email_summary import build_email_preview
from backend.sop.engine import SOPEngine
from backend.sop.spec import Phase
from backend.trace import TraceWriter
from tests.fakes import ScriptedProvider

TODAY = date(2026, 10, 2)  # after CL-2048's appeal deadline (2026-03-18)

MARGARET_OPENING = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my "
    "denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)
MARGARET_EXTRACTION = {
    "full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472",
    "policy_number": "POL-9921", "caller_role": "policyholder", "case_type": "healthcare",
    "case_status": "denied", "case_month": 1, "intent": "denial_question",
}


def run(extractions, messages, today=TODAY, reply=None):
    provider = ScriptedProvider(extractions, reply=reply)
    engine = SOPEngine(provider, today=lambda: today, tracer=TraceWriter(enabled=False))
    state = engine.start()
    results = [engine.handle(state, m) for m in messages]
    return state, results, provider


def assert_no_claim_data_before_verification(results):
    for r in results:
        if not r.trace["state"]["verified"]:
            assert r.trace["plan"]["facts"] == []
            assert "CL-" not in r.reply and "$" not in r.reply


# --- The demo test case -------------------------------------------------------------


def test_margaret_full_workflow():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"wants_to_end": True}, {"email_consent": "yes"}],
        [MARGARET_OPENING, "Yes, that's the one.", "No, that's all.", "Yes please."],
    )
    verify, confirm, wrap, close = results

    # Verified in one turn, the remembered hint resolves the claim without asking from scratch.
    assert verify.trace["phase_before"] == "VERIFY_ID"
    assert verify.trace["phase_after"] == "RESOLVE_INTENT"
    assert verify.trace["plan"]["action"] == "confirm_claim"
    assert "CL-2048" in verify.reply  # not CL-2011, the other January healthcare claim

    # PROCESS_CASE answers from grounded data; deadline passed -> human offered.
    assert confirm.trace["phase_after"] == "PROCESS_CASE"
    assert "pathology report" in confirm.reply
    assert "already passed" in confirm.reply
    assert confirm.reply.endswith("human claims representative to review your options?")
    assert confirm.trace["plan"]["pending_question"] == "offer_human"

    assert wrap.trace["phase_after"] == "POST_PROCESS"
    assert "Preview email" in wrap.reply

    assert state.phase == Phase.ENDED and state.email.consent is True
    assert "sent the summary" in close.reply


def test_hint_is_remembered_but_not_acted_on_before_verification():
    state, results, _ = run(
        [{"full_name": "Margaret Chen", "case_type": "healthcare", "case_status": "denied", "case_month": 1}],
        ["Margaret Chen here, calling about my denied healthcare claim from January"],
    )
    assert state.phase == Phase.VERIFY_ID
    assert state.hints.case_type == "healthcare" and state.hints.month == 1
    assert "denied healthcare claim from January" in results[0].trace["plan"]["directive"]
    assert_no_claim_data_before_verification(results)


# --- Verification gate --------------------------------------------------------------


def test_two_fields_plus_policy_number_is_not_enough():
    state, results, _ = run(
        [{"full_name": "Margaret Chen", "dob": "1985-03-15", "policy_number": "POL-9921"}],
        ["Margaret Chen, born 1985-03-15, policy POL-9921"],
    )
    assert state.phase == Phase.VERIFY_ID and not state.verified
    assert state.counters.verification_failures == 0  # not a failed attempt, just incomplete
    assert_no_claim_data_before_verification(results)


def test_mismatch_never_reveals_which_field_failed_and_escalates_after_three():
    wrong = {"full_name": "Margaret Chen", "dob": "1985-03-15"}
    state, results, _ = run(
        [{**wrong, "id_last4": "1111"}, {"id_last4": "2222"}, {"id_last4": "3333"}],
        ["Margaret Chen, 1985-03-15, last four 1111", "last four 2222", "last four 3333"],
    )
    first = results[0]
    assert first.trace["plan"]["action"] == "verification_mismatch"
    for word in ("ssn", "date of birth was", "name didn't", "incorrect"):
        assert word not in first.reply.lower()
    assert state.phase == Phase.ESCALATED and state.escalation_reason == "verification_failed"
    assert_no_claim_data_before_verification(results)


def test_frustrated_caller_gets_empathy_not_claim_details():
    state, results, _ = run(
        [
            {"full_name": "Margaret Chen"},
            {"emotion": "frustrated", "intent": "denial_question"},
        ],
        [
            "Margaret Chen",
            "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
        ],
    )
    plan = results[1].trace["plan"]
    assert plan["action"] == "ask_identity"
    assert "protected" in plan["directive"] and "generic acknowledgment" in plan["directive"]
    assert state.counters.frustration == 1 and state.phase == Phase.VERIFY_ID
    assert_no_claim_data_before_verification(results)


def test_repeated_frustration_escalates():
    upset = {"emotion": "angry", "refuses_to_share": True}
    state, _, _ = run([upset] * 3, ["No."] * 3)
    assert state.phase == Phase.ESCALATED and state.escalation_reason == "frustration_limit"


# --- Scope --------------------------------------------------------------------------


def test_off_topic_declined_then_escalated():
    state, results, _ = run([{"off_topic": True}] * 3, ["What is RL?"] * 3)
    assert results[0].trace["plan"]["action"] == "decline_off_topic"
    assert "only help with questions about your insurance claims" in results[0].reply
    assert state.phase == Phase.ESCALATED and state.escalation_reason == "off_topic_limit"


def test_conversation_stays_closed_after_escalation():
    state, results, provider = run([{"off_topic": True}] * 3, ["What is RL?"] * 3 + ["hello?"])
    assert results[-1].trace["plan"]["action"] == "conversation_over"


# --- Representatives ----------------------------------------------------------------


DAVID = {**MARGARET_EXTRACTION, "caller_role": "representative", "representative_name": "David Chen"}
DAVID_TEXT = "I'm David Chen calling for my mother Margaret Chen, DOB 1985-03-15, SSN last four 4472"


def start_david(extra_extractions=(), extra_messages=(), reply=None):
    provider = ScriptedProvider([DAVID, *extra_extractions], reply=reply)
    engine = SOPEngine(provider, today=lambda: TODAY, tracer=TraceWriter(enabled=False))
    state = engine.start()
    results = [engine.handle(state, m) for m in [DAVID_TEXT, *extra_messages]]
    return engine, state, results, provider


def test_representative_triggers_a_consent_request_not_access():
    _, state, results, _ = start_david()
    r = results[0]
    assert r.trace["plan"]["action"] == "consent_requested"
    assert "I've sent a consent request to the phone number on file (***-***-2836)" in r.reply
    assert not state.verified and state.consent.status == "pending"
    assert_no_claim_data_before_verification(results)


def test_caller_saying_she_approved_does_not_grant_access():
    _, state, results, _ = start_david([{"confirms_case": "yes"}], ["She approved it, go ahead"])
    assert results[1].trace["plan"]["action"] == "consent_waiting"
    assert not state.verified
    assert_no_claim_data_before_verification(results)


def test_policyholder_approval_continues_with_the_remembered_claim():
    engine, state, _, provider = start_david(reply="Thanks for waiting, David.")
    result = engine.consent_decision(state, "approved")
    assert state.verified_party_id == "P9" and state.phase == Phase.RESOLVE_INTENT
    # David mentioned the denied January healthcare claim -> confirm it, not ask from scratch.
    assert result.trace["plan"]["action"] == "confirm_claim"
    assert "CL-2048" in result.reply
    # Talking to David about Margaret: both names allowed, David addressed.
    assert "speaking with David Chen (son)" in provider.responder_systems[-1]
    assert state.transcript[-2].role == "event"


def test_policyholder_denial_offers_a_human_then_closes():
    engine, state, _, _ = start_david()
    result = engine.consent_decision(state, "denied")
    assert result.trace["plan"]["action"] == "offer_human"
    assert "Margaret didn't approve the request" in result.reply
    assert not state.verified
    for answer in ("no", "no"):
        engine.provider.extractions.append({"confirms_case": "no"})
        last = engine.handle(state, answer)
    assert state.phase == Phase.ENDED and "contact us directly" in last.reply


def test_no_response_button_and_nobody_clicking_both_stop_waiting():
    engine, state, _, _ = start_david()
    assert "haven't received a response" in engine.consent_decision(state, "no_response").reply

    # Nobody clicks: after the fixture's timeout length (5), the agent stops waiting.
    _, state, results, _ = start_david([{}] * 5, ["hello?"] * 5)
    actions = [r.trace["plan"]["action"] for r in results[1:]]
    assert actions == ["consent_waiting"] * 4 + ["offer_human"]
    assert state.consent.status == "no_response" and not state.verified


def test_consent_decision_is_rejected_when_nothing_is_pending():
    import pytest

    state, _, _ = run([MARGARET_EXTRACTION], [MARGARET_OPENING])
    engine = SOPEngine(ScriptedProvider([]), today=lambda: TODAY, tracer=TraceWriter(enabled=False))
    with pytest.raises(ValueError):
        engine.consent_decision(state, "approved")


def test_unlisted_representative_is_transferred():
    state, _, _ = run(
        [{**MARGARET_EXTRACTION, "caller_role": "representative", "representative_name": "Sam Smith"}],
        ["I'm Sam Smith calling for Margaret Chen, DOB 1985-03-15, SSN last four 4472"],
    )
    assert state.escalation_reason == "representative_not_authorized" and not state.verified


# --- Grounding ----------------------------------------------------------------------


def test_ungrounded_reply_is_replaced_by_fallback():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}],
        [MARGARET_OPENING, "yes"],
        reply="Good news, we will pay you $9,999.00 for claim CL-2048!",
    )
    answer = results[1]
    assert answer.trace["response"]["source"] == "fallback"
    assert "$9,999" not in answer.reply


def test_deadline_not_passed_gives_document_next_steps():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}],
        [MARGARET_OPENING, "yes"],
        today=date(2026, 2, 1),
    )
    assert "before March 18, 2026" in results[1].reply
    assert state.pending_question == "anything_else"


def test_picking_from_a_list_and_switching_claims():
    state, results, _ = run(
        [
            {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"},
            {"case_type": "auto", "intent": "status_inquiry"},
            {"confirms_case": "yes"},
            {"case_type": "dental", "intent": "payment_question"},
            {"confirms_case": "yes"},
        ],
        ["Margaret Chen, 1985-03-15, ssn 4472", "the auto one", "yes", "what about my dental claim?", "yes"],
    )
    # No hints yet -> list claims; picking "the auto one" from the list is confirmed first.
    assert results[0].trace["plan"]["action"] == "ask_which_claim"
    assert results[1].trace["plan"]["action"] == "confirm_claim"
    assert results[2].trace["state"]["selected_case_id"] == "CL-2102"
    # Switching to another claim is confirmed too.
    assert results[3].trace["phase_after"] == "RESOLVE_INTENT"
    assert results[3].trace["plan"]["action"] == "confirm_claim"
    assert state.selected_case_id == "CL-1899" and "$425.00" in results[4].reply


# --- Regression: yes/no answered the wrong question (live test, 2026-10-02) ---------


VERIFIED = {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}


def test_regression_choosing_by_month_confirms_instead_of_answering():
    # Caller picked "January 22th" from a list of the 2026 claims. Before the fix the
    # code silently selected CL-2048 and planned an answer, while the model asked a
    # confirmation question; the next "yes" was then read as accepting a transfer.
    state, results, _ = run(
        [
            VERIFIED,
            {"case_year": 2026, "intent": "status_inquiry"},
            {"case_month": 1},
            {"confirms_case": "yes", "case_type": "healthcare", "case_month": 1, "case_year": 2026},
        ],
        ["Margaret Chen, 1985-03-15, ssn 4472", "my 2026 ones", "Janurary 22th plz", "yes that one"],
    )
    pick, confirm = results[2], results[3]
    assert pick.trace["plan"]["action"] == "confirm_claim"
    assert pick.reply.endswith("Just to confirm, are you calling about claim CL-2048, a healthcare claim filed on January 12, 2026 (status: denied)?")
    # "yes that one" confirms the claim; it is NOT a request for a human.
    assert confirm.trace["phase_after"] == "PROCESS_CASE"
    assert confirm.trace["plan"]["action"] == "answer_case"
    assert not state.handoff_requested


def test_regression_question_about_transfer_is_answered_not_escalated():
    state, results, _ = run(
        [
            MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"confirms_case": "yes"},
            {"intent": "general_claim_question", "emotion": "confused"},
        ],
        [MARGARET_OPENING, "yes", "yes please", "Why do I have to talk to a human here?"],
    )
    accept, why = results[2], results[3]
    assert accept.trace["phase_after"] == "POST_PROCESS" and state.handoff_requested
    assert why.trace["plan"]["action"] == "ask_email_again"
    assert state.phase == Phase.POST_PROCESS  # still waiting on the email question
    assert any("already passed" in f for f in why.trace["plan"]["facts"])  # can explain why


def test_explicit_human_request_during_wrap_up_still_asks_email_first():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"wants_to_end": True}, {"wants_human": True}, {"email_consent": "no"}],
        [MARGARET_OPENING, "yes", "that's all", "actually get me a person", "no"],
    )
    assert results[3].trace["plan"]["action"] == "ask_email_again"
    assert state.phase == Phase.ESCALATED and state.email.consent is False


def test_declining_the_human_offer_twice_continues_normally():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"confirms_case": "no"},
         {"confirms_case": "no"}, {"confirms_case": "no"}],
        [MARGARET_OPENING, "yes", "no thanks", "no", "no"],
    )
    assert results[2].trace["plan"]["action"] == "reoffer_human"  # one gentle re-offer
    assert results[3].trace["plan"]["action"] == "human_offer_declined"
    assert results[3].reply.endswith("Is there anything else I can help you with?")
    assert results[4].trace["phase_after"] == "POST_PROCESS"  # "no" to "anything else?"
    assert not state.handoff_requested


def test_model_cannot_add_its_own_question():
    state, results, _ = run(
        [MARGARET_EXTRACTION],
        [MARGARET_OPENING],
        reply="Thanks, Margaret, you're verified. Is that the one you're asking about?",
    )
    reply = results[0].reply
    assert reply.count("?") == 1 and reply.endswith("(status: denied)?")
    assert results[0].trace["response"]["source"] == "llm"


# --- Regression: verified caller with no claims, new-claim request (live test) -------

AVA = {"full_name": "Ava Martinez Lopez", "dob": "1990-08-21", "id_last4": "9180", "email": "ava.lopez@email.com"}
AVA_TEXT = "Ava Martinez Lopez, 1990 21st august, 9180, ava.lopez@email.com"


def test_regression_no_claims_on_file_is_said_plainly():
    state, results, _ = run([AVA, {"case_type": "healthcare"}], [AVA_TEXT, "my healthcare claim"])
    verified, asked = results
    assert verified.trace["plan"]["action"] == "no_claims_on_file"
    assert verified.reply == (
        "Thank you for verifying your identity. I don't see any existing claims with us. "
        "What can I help you with today?"
    )
    # Asking about a claim anyway -> nothing to look up, offer a human.
    assert asked.trace["plan"]["action"] == "offer_human"
    assert "don't see any claims on file" in asked.reply and "Which claim" not in asked.reply
    assert state.verified and state.pending_question == "offer_human"


def test_no_claims_caller_with_nothing_else_closes():
    state, results, _ = run([AVA, {}], [AVA_TEXT, "nothing, thanks"])
    assert results[1].trace["plan"]["action"] == "close" and state.phase == Phase.ENDED


def test_regression_new_claim_request_flow_from_the_conversation():
    # Your example: offer -> "no I want you to do it" -> re-offer -> "NO" -> anything else.
    state, results, _ = run(
        [
            AVA,
            {"unsupported_request": "filing a new claim", "case_type": "healthcare"},
            {"unsupported_request": "filing a new claim", "confirms_case": "no"},
            {"confirms_case": "no"},
            {"confirms_case": "no"},
        ],
        [AVA_TEXT, "I want to file a new healthcare claim", "no I want you to do it", "NO", "no"],
    )
    offer, reoffer, declined, close = results[1:]
    assert offer.trace["plan"]["action"] == "offer_human"
    assert "not able to help with filing a new claim" in offer.reply
    assert reoffer.trace["plan"]["action"] == "reoffer_human"
    assert "recommended way" in reoffer.reply
    assert declined.reply.endswith("Is there anything else I can help you with?")
    assert close.trace["plan"]["action"] == "close" and state.phase == Phase.ENDED


def test_accepting_human_with_nothing_discussed_connects_immediately():
    state, results, _ = run(
        [AVA, {"unsupported_request": "filing a new claim"}, {"confirms_case": "yes"}],
        [AVA_TEXT, "I want to file a new claim", "Yes, connect me to one"],
    )
    assert state.phase == Phase.ESCALATED and not state.email.offered
    assert "connecting you" in results[-1].reply


def test_accepting_human_after_a_claim_was_discussed_offers_email_first():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"confirms_case": "yes"}, {"email_consent": "yes"}],
        [MARGARET_OPENING, "yes", "yes connect me", "yes send it"],
    )
    assert results[2].trace["plan"]["action"] == "offer_email"
    assert "connect you with a human representative right after" in results[2].reply
    assert state.phase == Phase.ESCALATED and state.email.consent is True


def test_unsupported_request_before_verification_reveals_nothing():
    state, results, _ = run([{"unsupported_request": "filing a new claim"}], ["I want to file a new claim"])
    assert results[0].trace["plan"]["action"] == "offer_human"
    assert_no_claim_data_before_verification(results)


def test_regression_model_cannot_claim_a_transfer_that_did_not_happen():
    state, results, _ = run(
        [AVA, {"case_type": "healthcare"}],
        [AVA_TEXT, "my healthcare claim"],
        reply="I'm connecting you now to our claims filing team. One moment.",
    )
    r = results[1]
    assert r.trace["response"]["source"] == "fallback"
    assert "claims an action the workflow did not take" in r.trace["response"]["guard_violations"]
    assert state.phase == Phase.RESOLVE_INTENT  # nothing was transferred


def test_stuck_loop_offers_a_human_instead_of_repeating():
    # A caller who keeps answering without giving anything usable.
    state, results, _ = run([{}, {}, {}], ["hmm", "what", "ok"])
    asks = [r.trace["plan"]["action"] for r in results]
    assert asks == ["ask_identity", "ask_identity", "offer_human"]
    assert "stuck" in results[2].trace["plan"]["reasons"][0]


# --- Claims are never listed (feedback from live test) ------------------------------


def _no_claim_details(reply: str):
    assert "CL-" not in reply and "2026" not in reply and "2025" not in reply


def test_after_verification_claims_are_not_listed():
    _, results, _ = run([VERIFIED], ["Margaret Chen, 1985-03-15, ssn 4472"])
    r = results[0]
    assert r.trace["plan"]["action"] == "ask_which_claim"
    assert r.trace["plan"]["facts"] == []  # the model isn't even given the claims
    assert r.reply == "Thank you, you're verified. I see you have some claims with us. Which claim are you calling about?"


def test_ambiguous_description_asks_for_a_distinguishing_detail():
    state, results, _ = run(
        [VERIFIED, {"case_type": "healthcare"}, {"case_status": "denied"}, {"confirms_case": "yes"}],
        ["Margaret Chen, 1985-03-15, ssn 4472", "my healthcare claim", "the denied one", "yes"],
    )
    narrow = results[1]
    assert narrow.trace["plan"]["action"] == "narrow_claim"
    assert narrow.reply == (
        "I see more than one healthcare claim on your account. To find the right one, could you "
        "tell me when it was filed, its status, or the claim number?"
    )
    _no_claim_details(narrow.reply)
    assert results[2].trace["plan"]["action"] == "confirm_claim"
    assert state.selected_case_id == "CL-2048"


def test_claim_can_be_identified_by_filing_date():
    state, results, _ = run(
        [VERIFIED, {"case_month": 1, "case_day": "28th", "case_year": 2025}, {"confirms_case": "yes"}],
        ["Margaret Chen, 1985-03-15, ssn 4472", "the one I filed on January 28th 2025", "yes"],
    )
    assert "CL-2011" in results[1].reply and state.selected_case_id == "CL-2011"


def test_no_match_does_not_reveal_other_claims():
    _, results, _ = run(
        [VERIFIED, {"case_type": "life"}],
        ["Margaret Chen, 1985-03-15, ssn 4472", "my life insurance claim"],
    )
    r = results[1]
    assert r.trace["plan"]["action"] == "no_matching_claim"
    assert "couldn't find a life claim" in r.reply
    _no_claim_details(r.reply)


# --- Regression: wrong name after another identity was mentioned (live test) ---------

YAVEN = {"full_name": "Yaven Li"}
YAVEN_REST = {"id_last4": "5317", "dob": "1989-12-03"}
MARGARET_IDENTITY = {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}


def test_regression_wrong_name_is_retried_once():
    state, results, provider = run(
        [YAVEN, YAVEN_REST, MARGARET_IDENTITY],
        ["Hello I am yaven li", "5317, 1989-12-03", MARGARET_OPENING],
        reply=[
            "Thanks, Yaven. I just need a bit more.",  # unverified: no names allowed
            "Thanks. I just need a bit more.",  # retry OK
            "Thank you, Ya Wen. I don't see any existing claims with us.",  # verified name: OK
            "I appreciate those details, Margaret.",  # wrong person
            "I appreciate those details, Ya Wen.",  # retry OK
        ],
    )
    first, verified, switched = results
    assert first.trace["response"]["retried"] and first.trace["response"]["source"] == "llm"
    assert "Yaven" not in first.reply
    assert not verified.trace["response"]["retried"]
    assert switched.trace["response"]["retried"] and "Ya Wen" in switched.reply
    assert "Margaret" not in switched.reply
    assert "The verified customer is Ya Wen Li" in provider.responder_systems[-1]
    assert state.verified_party_id == "P13"  # the session never switches accounts


def test_wrong_name_twice_falls_back_to_a_reply_without_names():
    state, results, _ = run(
        [YAVEN, YAVEN_REST, MARGARET_IDENTITY],
        ["Hello I am yaven li", "5317, 1989-12-03", MARGARET_OPENING],
        reply=["Okay.", "Okay.", "Hello, Margaret.", "Hello again, Margaret."],
    )
    switched = results[2]
    assert switched.trace["response"]["source"] == "fallback"
    assert switched.trace["response"]["retried"]
    assert "Margaret" not in switched.reply


# --- Regression: "send it to a different email" (live test) --------------------------

TO_WRAP_UP_WITH_HANDOFF = [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"confirms_case": "yes"}]
TO_WRAP_UP_TEXT = [MARGARET_OPENING, "Yes", "sure"]


def test_regression_different_email_is_not_a_yes_and_is_never_used():
    state, results, _ = run(
        TO_WRAP_UP_WITH_HANDOFF + [
            {"email_consent": "yes", "unsupported_request": "updating your email address"},
            {"email_consent": "yes"},
        ],
        TO_WRAP_UP_TEXT + [
            "Yes i like that, can u send it to a different email thou? the one on file cant be accessed by me",
            "ok yes send it there",
        ],
        # Even if the model offers to take a new address, it must never reach the caller.
        reply="I can help with that, Margaret. What's the email address you'd like me to use instead?",
    )
    different, final = results[3], results[4]
    assert different.trace["plan"]["action"] == "email_on_file_only"
    assert different.trace["state"]["email"]["consent"] is None  # "yes, but elsewhere" isn't consent
    assert "What's the email address" not in different.reply  # model's question stripped
    assert different.reply.endswith("to m***@email.com?")
    # Final turn: code's exact text, no question, transfer happens.
    assert final.reply == (
        "I've sent the summary to m***@email.com. I'm connecting you with a human "
        "representative now. Please hold for a moment."
    )
    assert state.phase == Phase.ESCALATED and state.email.consent is True


def test_typed_email_not_on_file_is_treated_as_a_change_request():
    _, results, _ = run(
        TO_WRAP_UP_WITH_HANDOFF + [{"email_consent": "yes", "email": "maggie.new@gmail.com"}],
        TO_WRAP_UP_TEXT + ["yes, send it to maggie.new@gmail.com"],
    )
    assert results[3].trace["plan"]["action"] == "email_on_file_only"


def test_personal_detail_request_without_handoff_offers_a_human_then_back_to_email():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"wants_to_end": True},
         {"unsupported_request": "updating your email address"},
         {"confirms_case": "no"}, {"confirms_case": "no"}, {"email_consent": "no"}],
        [MARGARET_OPENING, "yes", "that's all", "send it to my new email instead",
         "no", "no", "no thanks"],
    )
    assert results[3].trace["plan"]["action"] == "offer_human"
    assert "only send the summary to the email address on file" in results[3].reply
    assert results[4].trace["plan"]["action"] == "reoffer_human"
    assert results[5].reply.endswith("to m***@email.com?")  # back to the email question
    assert state.phase == Phase.ENDED and state.email.consent is False


def test_update_request_mid_call_offers_a_human():
    _, results, _ = run(
        [MARGARET_EXTRACTION, {"unsupported_request": "updating your phone number"}],
        [MARGARET_OPENING, "can you update my phone number?"],
    )
    assert results[1].trace["plan"]["action"] == "offer_human"
    assert "not able to help with updating your phone number" in results[1].reply


def test_final_messages_are_never_paraphrased():
    state, results, _ = run(
        [{"off_topic": True}] * 3, ["What is RL?"] * 3,
        reply="Sure! Anything else I can do? Maybe tell me about RL?",
    )
    assert results[-1].trace["response"]["source"] == "fallback"
    assert results[-1].reply.endswith("Please hold for a moment.")


# --- Emotions: each handled differently; no empathy when none was expressed ---------


def test_regression_no_unprompted_frustration_on_a_neutral_message():
    # Live test: "The healthcare one!!!" (neutral) got "I hear your frustration".
    state, results, _ = run(
        [VERIFIED, {"case_type": "healthcare"}],
        ["Margaret Chen, 1985-03-15, ssn 4472", "The healthcare one!!!"],
        reply=[
            "Thanks, you're verified.",
            "I hear your frustration. I see more than one healthcare claim.",  # not allowed
            "I see more than one healthcare claim on your account.",  # retry OK
        ],
    )
    r = results[1]
    assert r.trace["response"]["retried"] and r.trace["response"]["source"] == "llm"
    assert "frustration" not in r.reply


def test_empathy_is_allowed_when_the_emotion_was_detected():
    _, results, _ = run(
        [{"full_name": "Margaret Chen"}, {"emotion": "frustrated"}],
        ["Margaret Chen", "This is ridiculous, I already told you who I am"],
        reply=["Thanks.", "I know this has taken a few steps, and I'm sorry it's been frustrating."],
    )
    assert results[1].trace["response"]["source"] == "llm"
    assert not results[1].trace["response"]["retried"]


def _directive_for(emotion: str) -> str:
    _, results, _ = run([{"full_name": "Margaret Chen", "emotion": emotion}], ["Margaret Chen"])
    return results[0].trace["plan"]["directive"]


def test_each_emotion_gets_its_own_guidance():
    assert "apology" in _directive_for("angry")
    assert "Reassure" in _directive_for("anxious")
    assert "plain words" in _directive_for("confused")
    assert "generic acknowledgment" in _directive_for("frustrated")
    # A first frustrated message must not be told "we've been going back and forth".
    assert "Do not describe the conversation itself" in _directive_for("frustrated")
    assert "sounds" not in _directive_for("neutral")


def test_anxious_callers_hear_why_verification_matters():
    assert "protected" in _directive_for("anxious")


def test_anxiety_and_confusion_never_count_toward_a_transfer():
    state, _, _ = run(
        [{"emotion": "anxious"}, {"emotion": "confused"}, {"emotion": "anxious"}],
        ["is my data safe?", "what do you mean?", "I'm worried"],
    )
    assert state.counters.frustration == 0 and state.phase != Phase.ESCALATED


def test_email_preview_contains_discussion_outcome_and_next_steps():
    state, _, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"wants_to_end": True}],
        [MARGARET_OPENING, "yes", "that's all"],
    )
    from backend.data import get_data

    preview = build_email_preview(state, get_data())
    assert preview.to == "margaret@email.com"
    assert "CL-2048" in preview.body and "denied" in preview.body
    assert "human claims representative" in preview.body

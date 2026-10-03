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
    assert "protected" in plan["directive"] and "acknowledging" in plan["directive"]
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


def test_representative_is_not_given_access_without_consent():
    state, results, _ = run(
        [{**MARGARET_EXTRACTION, "caller_role": "representative", "representative_name": "David Chen"}],
        ["I'm David Chen calling for my mother Margaret Chen, DOB 1985-03-15, SSN last four 4472"],
    )
    assert not state.verified
    assert state.escalation_reason == "representative_consent_required"
    assert_no_claim_data_before_verification(results)


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
    for r in results:
        assert r.trace["plan"]["action"] == "offer_human"
        assert "don't see any claims" in r.reply and "these claims" not in r.reply
        assert "Which claim" not in r.reply
    assert state.verified and state.pending_question == "offer_human"


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

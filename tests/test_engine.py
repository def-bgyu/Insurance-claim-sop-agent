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
    assert state.human_offered

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
    assert not state.human_offered


def test_picking_from_a_list_and_switching_claims():
    state, results, _ = run(
        [
            {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"},
            {"case_type": "auto", "intent": "status_inquiry"},
            {"case_type": "dental", "intent": "payment_question"},
            {"confirms_case": "yes"},
        ],
        ["Margaret Chen, 1985-03-15, ssn 4472", "the auto one", "what about my dental claim?", "yes"],
    )
    # No hints yet -> list claims; "the auto one" picks from the list directly.
    assert results[0].trace["plan"]["action"] == "ask_which_claim"
    assert results[1].trace["state"]["selected_case_id"] == "CL-2102"
    # Switching to a claim inferred from a description is confirmed first.
    assert results[2].trace["phase_after"] == "RESOLVE_INTENT"
    assert results[2].trace["plan"]["action"] == "confirm_claim"
    assert state.selected_case_id == "CL-1899" and "$425.00" in results[3].reply


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

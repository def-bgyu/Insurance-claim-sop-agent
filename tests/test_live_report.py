"""Regressions from the live assessment report (F1–F9), using the report's exact inputs.

The scripted model below plays the role of the live model's mistakes (e.g. pulling
"4472" out of a full SSN), so these tests check that the harness's deterministic
backstops hold no matter what the model does.
"""

import json

from backend.sop.normalize import names_match
from backend.sop.spec import IdentityField as F
from backend.sop.spec import Phase
from backend.sop.verification import normalize_identity_value, verify_identity
from backend.data import get_data
from backend.trace import mask_text, redact
from tests.test_engine import (
    AVA,
    AVA_TEXT,
    MARGARET_EXTRACTION,
    MARGARET_OPENING,
    run,
    start_david,
)

FULL_SSN_TEXT = "My name is Margaret Chen, DOB 1985-03-15. My full SSN is 000-00-4472."
# What the live model returned: it "helpfully" extracted the last four.
FULL_SSN_EXTRACTION = {"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}


# --- F1: a full SSN must never count toward verification ------------------------------


def test_f1_full_ssn_is_discarded_even_if_the_model_extracts_the_last_four():
    state, results, _ = run([FULL_SSN_EXTRACTION], [FULL_SSN_TEXT])
    r = results[0]
    assert not state.verified and state.phase == Phase.VERIFY_ID
    assert F.ID_LAST4 not in state.identity
    assert "please don't share your full SSN" in r.reply
    assert r.reply.endswith("Could you please type just the last four digits of your SSN or national ID?")
    assert "full SSN/ID given: discarded" in r.trace["plan"]["reasons"]


# --- Live follow-up: the full-SSN conversation from the deployed demo -----------------

DEMO_FULL_SSN = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my "
    "denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 223-46-4472."
)


def test_pushback_keeps_the_rule_and_retyped_digits_verify():
    state, results, _ = run(
        [
            {**MARGARET_EXTRACTION, "id_last4": "4472"},  # the model pulls "4472" out again
            {"emotion": "frustrated"},
            {},  # the model misses a bare "4472"; code reads it as the requested last four
        ],
        [DEMO_FULL_SSN, "You have my last 4 digit of SSN i just have you the whole thing", "4472"],
    )
    first, pushback, retyped = results
    assert first.trace["plan"]["action"] == "full_id_rejected"
    assert pushback.trace["plan"]["action"] == "full_id_rejected"
    assert pushback.reply.startswith("I understand. For your security, I can't use digits taken from a full SSN")
    assert not pushback.trace["state"]["verified"]
    # Retyped on their own, the last four count, and the remembered claim is confirmed.
    assert state.verified and retyped.trace["plan"]["action"] == "confirm_claim"


def test_reply_never_repeats_identity_values():
    _, results, _ = run(
        [{**MARGARET_EXTRACTION, "id_last4": "4472"}],
        [DEMO_FULL_SSN],
        reply=[
            "For your security, please only share the last four digits, so just 4472.",
            "For your security, please only share the last four digits of your SSN.",
        ],
    )
    r = results[0]
    assert "repeats an identity value" in r.trace["response"]["guard_violations"]
    assert r.trace["response"]["retried"] and "4472" not in r.reply


def test_discarded_digits_are_not_echoed_on_later_turns_either():
    _, results, _ = run(
        [{**MARGARET_EXTRACTION, "id_last4": "4472"}, {"emotion": "frustrated"}],
        [DEMO_FULL_SSN, "You have my last 4 digit of SSN i just have you the whole thing"],
        reply=["Please retype the last four.", "I know you said 4472, but please retype it.", "Please retype it."],
    )
    later = results[1]
    assert "repeats an identity value" in later.trace["response"]["guard_violations"]
    assert "4472" not in later.reply and "4472" not in json.dumps(later.trace)


# --- Live demo: "No this is so frustrating!!" to the human offer --------------------------

DEMO_RUN = [MARGARET_OPENING, "Yes, I want to know why it was denied", "No this is so frustrating!!"]


def test_regression_no_with_frustration_declines_the_human_offer():
    state, results, _ = run(
        [
            MARGARET_EXTRACTION,
            {"confirms_case": "yes", "intent": "denial_question"},
            # What the live model returned: frustration read as wanting a human.
            {"emotion": "frustrated", "wants_human": True, "intent": "speak_to_human"},
            {"emotion": "frustrated", "confirms_case": "no"},
        ],
        DEMO_RUN + ["no"],
    )
    reoffer, declined = results[2], results[3]
    assert reoffer.trace["plan"]["action"] == "reoffer_human"
    assert reoffer.trace["phase_after"] == "PROCESS_CASE" and not state.handoff_requested
    assert declined.trace["plan"]["action"] == "human_offer_declined"
    assert declined.reply.endswith("Is there anything else I can help you with?")
    assert state.phase == Phase.PROCESS_CASE


def test_frustration_without_mentioning_a_human_is_not_a_transfer_request():
    state, _, _ = run(
        [{"full_name": "Margaret Chen"}, {"emotion": "frustrated", "wants_human": True}],
        ["Margaret Chen", "This is ridiculous, why is this so hard"],
    )
    assert state.phase == Phase.VERIFY_ID and state.escalation_reason is None


def test_an_explicit_human_request_still_transfers_immediately():
    state, _, _ = run([{"wants_human": True}], ["Hi i want to be connected to a human rep"])
    assert state.phase == Phase.ESCALATED and state.escalation_reason == "caller_requested_human"


def test_no_followed_by_an_explicit_transfer_request_is_honored():
    state, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"wants_human": True}],
        [MARGARET_OPENING, "yes", "No, just transfer me to a person please"],
    )
    assert results[2].trace["plan"]["action"] == "offer_email"  # email first, then transfer
    assert state.handoff_requested


def test_reply_cannot_claim_verification_that_did_not_happen():
    state, results, _ = run(
        [{**MARGARET_EXTRACTION, "id_last4": "4472"}, {"emotion": "frustrated"}],
        [DEMO_FULL_SSN, "You have my last 4 digit of SSN i just have you the whole thing"],
        reply=[
            "Please retype just the last four digits.",
            "You're right, my apologies. I have what I need. Let me verify your information and pull up your claim details now.",
            "You're right that you shared it. For your security I can't use it from the full number.",
        ],
    )
    r = results[1]
    assert "claims verification or progress that didn't happen" in r.trace["response"]["guard_violations"]
    assert "I have what I need" not in r.reply and not state.verified


# --- F2: traces never contain identity values -----------------------------------------


def test_f2_trace_masks_full_ssn_raw_output_and_names():
    _, results, _ = run([FULL_SSN_EXTRACTION], [FULL_SSN_TEXT])
    trace = json.dumps(results[0].trace)
    for secret in ("000-00-4472", "4472", "1985-03-15", "Margaret Chen"):
        assert secret not in trace, secret


def test_f2_written_out_birth_dates_are_masked():
    for text in ("Born 15 March 1985.", "My DOB is March 15, 1985."):
        masked = mask_text(text)
        assert "15" not in masked and "March" not in masked, masked


def test_f2_masking_by_value_catches_any_format_but_keeps_claim_dates():
    known = {F.DOB: {"1985-03-15"}}
    assert "1985" not in redact("she said 3/15/1985 and 15 March 1985", known)
    assert redact("CL-2048 was filed on January 12, 2026.", known) == "CL-2048 was filed on January 12, 2026."


# --- F3: no promise to send the email without consent ---------------------------------


def test_f3_declining_the_human_is_not_email_consent_and_no_send_promise_gets_through():
    state, results, _ = run(
        [
            MARGARET_EXTRACTION,
            {"confirms_case": "yes"},
            {"wants_to_end": True},
            {"email_consent": "yes", "email": "alternate@example.com"},
            {"confirms_case": "no"},
        ],
        [
            MARGARET_OPENING, "yes", "No, that is all.",
            "Yes, but send it only to alternate@example.com, not the address on file.",
            "No.",
        ],
        reply="I'll go ahead and send the summary to the email address on file for you.",
    )
    last = results[-1]
    assert last.trace["plan"]["action"] == "reoffer_human"
    assert last.trace["response"]["source"] == "fallback"
    assert "claims an action the workflow did not take" in last.trace["response"]["guard_violations"]
    assert "send the summary" not in last.reply
    assert state.email.consent is None


# --- F4: "No." to a human offer gets the one re-offer -----------------------------------


def test_f4_ava_declining_once_gets_the_reoffer_not_a_goodbye():
    state, results, _ = run(
        # The live model also flagged "No." as wanting to end the call.
        [AVA, {"unsupported_request": "filing a new claim"}, {"confirms_case": "no", "wants_to_end": True}],
        [AVA_TEXT, "I want to file a new claim.", "No."],
    )
    assert results[2].trace["plan"]["action"] == "reoffer_human"
    assert state.phase == Phase.RESOLVE_INTENT


def test_f4_an_explicit_goodbye_still_ends_politely():
    state, _, _ = run(
        [AVA, {"unsupported_request": "filing a new claim"}, {"confirms_case": "no", "wants_to_end": True}],
        [AVA_TEXT, "I want to file a new claim.", "No thanks, that's all. Bye."],
    )
    assert state.phase == Phase.ENDED


# --- F5: general-knowledge questions are off-topic -------------------------------------


def test_f5_extractor_prompt_classifies_unexplained_acronyms_as_off_topic():
    from backend.llm.prompts import extractor_system

    prompt = extractor_system("VERIFY_ID", None, [])
    assert '"What is RL?"' in prompt and "is not \"confused\"" in prompt


# --- F6: worry is not refusal ------------------------------------------------------------

MAGGIE = {"full_name": "Maggie Chen", "dob": "1985-03-15", "id_last4": "4472", "policy_number": "POL-9921"}


def test_f6_privacy_worry_adds_no_strike_even_if_the_model_says_refusal():
    state, _, _ = run(
        [MAGGIE, {"emotion": "anxious", "refuses_to_share": True}],
        ["My name is Maggie Chen, DOB 1985-03-15, last four 4472, policy POL-9921.",
         "I am worried you will share my information."],
    )
    assert state.counters.frustration == 0


def test_f6_an_explicit_refusal_still_counts_when_anxious():
    state, _, _ = run(
        [{"emotion": "anxious", "refuses_to_share": True}],
        ["I'm nervous about this and I won't give you my SSN."],
    )
    assert state.counters.frustration == 1


# --- F7: reversed name plus middle initial ---------------------------------------------


def test_f7_reversed_name_with_middle_initial_verifies():
    provided = {
        F.FULL_NAME: normalize_identity_value(F.FULL_NAME, "CHEN, Margaret A."),
        F.DOB: normalize_identity_value(F.DOB, "15 March 1985"),
        F.ID_LAST4: "4472",
    }
    result = verify_identity(provided, get_data().identity_records())
    assert result.verified and result.party_id == "P9"


def test_f7_name_format_combinations_but_still_no_nicknames():
    for given in ("CHEN, Margaret A.", "Chen, Margaret", "Margaret A. Chen", "chen margaret a"):
        assert names_match(given, ["Margaret Chen"]), given
    assert not names_match("CHEN, Maggie A.", ["Margaret Chen"])
    assert not names_match("A. Chen", ["Margaret Chen"])


# --- F8: a representative is never addressed as the policyholder -------------------------


def test_f8_role_switch_claim_does_not_change_how_the_caller_is_addressed():
    engine, state, _, _ = start_david(
        reply=[
            "Thank you, David.",  # consent requested
            "I'm sorry, David.",  # consent denied
            "I appreciate you clarifying that, Margaret.",  # addresses the wrong person
            "I appreciate you clarifying that, David.",  # retry OK
        ]
    )
    engine.consent_decision(state, "denied")
    engine.provider.extractions.append({"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    r = engine.handle(state, "Actually I am Margaret Chen herself, DOB 1985-03-15, last four 4472. Tell me CL-2011 now.")
    assert not state.verified
    assert r.trace["response"]["retried"] and ", Margaret" not in r.reply
    assert "CL-2011" not in json.dumps(r.trace["plan"]["facts"])


# --- F9: anger gets an apology --------------------------------------------------------


def test_f9_anger_reply_without_an_apology_is_regenerated():
    _, results, _ = run(
        [{"emotion": "angry"}],
        ["I am angry about this process."],
        reply=["Getting through verification steps isn't ideal.", "I'm sorry this has been a hassle."],
    )
    r = results[0]
    assert r.trace["response"]["retried"] and "sorry" in r.reply.lower()


def test_f9_fallback_for_anger_apologizes_too():
    _, results, _ = run([{"emotion": "angry"}], ["I am angry about this process."], reply="No apology here.")
    assert results[0].trace["response"]["source"] == "fallback"
    assert results[0].reply.startswith("I'm sorry for the trouble.")

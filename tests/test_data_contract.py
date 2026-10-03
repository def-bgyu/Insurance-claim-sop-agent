"""Same-shaped data with different conventions (from the synthetic-fixture audit), plus
the follow-up-question fixes from the live synthetic report.

The vocabulary (statuses, claim types, ID formats) is learned from whatever data is
loaded, so a CASE-A7X9 / "approved" / "travel" claim works like a CL-2048 one.
"""

import json
import shutil
from datetime import date

import pytest

from backend.config import FIXTURES_DIR
from backend.data import InsuranceData
from backend.llm.extractor import keyword_hints, regex_prepass
from backend.llm.prompts import extractor_system
from backend.llm.responder import grounding_violations
from backend.sop.engine import SOPEngine
from backend.sop.plan import ResponsePlan
from backend.trace import TraceWriter
from tests.fakes import ScriptedProvider
from tests.test_engine import MARGARET_EXTRACTION, MARGARET_OPENING, VERIFIED, run


@pytest.fixture
def custom_data(tmp_path) -> InsuranceData:
    """The provided fixtures, with Margaret's auto claim and policy in new formats."""
    for f in FIXTURES_DIR.glob("*.json"):
        shutil.copy(f, tmp_path / f.name)
    claims = json.loads((tmp_path / "claims.json").read_text(encoding="utf-8"))
    for c in claims:
        if c["case_id"] == "CL-2102":
            c.update(case_id="CASE-A7X9", case_type="travel", status="approved")
    (tmp_path / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    holders = json.loads((tmp_path / "policyholders.json").read_text(encoding="utf-8"))
    for p in holders:
        if p["party_id"] == "P9":
            p["policy_number"] = "POLICY-A7X9"
    (tmp_path / "policyholders.json").write_text(json.dumps(holders), encoding="utf-8")
    return InsuranceData(tmp_path)


def run_on(data, extractions, messages, reply=None):
    engine = SOPEngine(
        ScriptedProvider(extractions, reply=reply), data=data,
        today=lambda: date(2026, 10, 2), tracer=TraceWriter(enabled=False),
    )
    state = engine.start()
    return state, [engine.handle(state, m) for m in messages]


# --- A: statuses and claim types come from the data ------------------------------------


def test_vocabulary_is_learned_from_the_loaded_data(custom_data):
    vocab = custom_data.vocabulary
    assert "approved" in vocab.statuses and "travel" in vocab.case_types
    prompt = extractor_system("RESOLVE_INTENT", None, [], list(vocab.statuses), list(vocab.case_types))
    assert '"approved"' in prompt and '"travel"' in prompt
    assert '"approved" is not "closed"' in prompt


def test_approved_travel_claim_is_found_by_description(custom_data):
    state, results = run_on(
        custom_data,
        [VERIFIED, {"case_type": "travel", "case_status": "approved"}],
        ["Margaret Chen, 1985-03-15, ssn 4472", "I need my approved travel claim"],
    )
    assert results[1].trace["plan"]["action"] == "confirm_claim"
    assert "CASE-A7X9" in results[1].reply


def test_keyword_backup_knows_the_data_s_statuses_and_types(custom_data):
    found = keyword_hints("I need my approved travel claim", custom_data.vocabulary)
    assert found["case_type"] == "travel" and found["case_status"] == "approved"
    # Everyday verbs aren't mistaken for a status.
    assert "case_status" not in keyword_hints("can you open the portal for me", custom_data.vocabulary)


# --- E: ID formats come from the data ---------------------------------------------------


def test_regex_backup_reads_ids_in_the_data_s_format(custom_data):
    found = regex_prepass("About case-a7x9 on policy POLICY-A7X9 please", custom_data.vocabulary)
    assert found["case_id"] == "CASE-A7X9" and found["policy_number"] == "POLICY-A7X9"


def test_grounding_guard_catches_invented_ids_in_any_data_format(custom_data):
    plan = ResponsePlan(
        action="answer_case", directive="", fallback_text="",
        facts=["Claim CASE-A7X9 is a travel claim."],
        claim_id_pattern=custom_data.vocabulary.claim_id_re,
    )
    assert grounding_violations("Your claim CASE-A7X9 is approved.", plan) == []
    assert grounding_violations("Your claim CASE-Z9Q8 is approved.", plan) == ["ungrounded claim id CASE-Z9Q8"]
    assert grounding_violations("See CL-99999.", plan) == ["ungrounded claim id CL-99999"]


def test_exact_claim_id_in_a_new_format_selects_the_claim(custom_data):
    state, results = run_on(
        custom_data, [VERIFIED, {"intent": "payment_question"}],
        ["Margaret Chen, 1985-03-15, ssn 4472", "what about CASE-A7X9?"],
    )
    assert state.selected_case_id == "CASE-A7X9"


# --- B: distinct follow-up questions are progress, not a loop ---------------------------

BEFORE_DEADLINE = date(2026, 2, 1)  # so CL-2048 answers end with "anything else?"


def test_regression_three_different_questions_are_not_stuck():
    state, results, _ = run(
        [
            MARGARET_EXTRACTION, {"confirms_case": "yes"},
            {"intent": "document_submission", "followup_topics": ["processing_time_after_submission"]},
            {"intent": "document_submission", "followup_topics": ["submission_method"]},
            {"intent": "document_submission", "followup_topics": ["missing_required_material_alternatives"]},
        ],
        [
            MARGARET_OPENING, "Yes.",
            "How long after I submit the missing documents?",
            "Where do I submit them, and what if I cannot get the documents?",
            "I cannot obtain either required document. What alternatives can I use?",
        ],
        today=BEFORE_DEADLINE,
    )
    assert [r.trace["plan"]["action"] for r in results[2:]] == ["answer_case"] * 3


def test_saying_nothing_new_three_times_is_still_stuck():
    state, results, _ = run([{}, {}, {}], ["hmm", "what", "ok"])
    assert results[2].trace["plan"]["action"] == "offer_human"


# --- C: a two-part document question gets both answers ----------------------------------


def test_regression_two_part_document_question_answers_both_parts():
    _, results, _ = run(
        [MARGARET_EXTRACTION, {"confirms_case": "yes"},
         # The model only picked the first part, as in the live test.
         {"intent": "document_submission", "followup_topics": ["submission_method"]}],
        [MARGARET_OPENING, "Yes.", "Where do I submit them, and what if I cannot get the documents?"],
        today=BEFORE_DEADLINE,
    )
    facts = " ".join(results[2].trace["plan"]["facts"])
    assert "member portal" in facts  # where to submit
    assert "is unavailable" in facts  # what to do without the documents

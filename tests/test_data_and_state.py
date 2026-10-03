import pytest

from backend.data import get_data
from backend.sop.spec import Phase
from backend.sop.state import SessionState

DATA = get_data()


# --- Data layer --------------------------------------------------------------------


def test_claims_are_scoped_to_their_owner():
    assert DATA.get_claim("P9", "CL-2048") is not None
    assert DATA.get_claim("P12", "CL-2048") is None  # exists, but not P12's


def test_margaret_has_two_january_healthcare_claims_but_one_denied():
    january_healthcare = [
        c for c in DATA.list_claims("P9") if c.case_type == "healthcare" and c.created_at.month == 1
    ]
    assert {c.case_id for c in january_healthcare} == {"CL-2048", "CL-2011"}
    assert [c.case_id for c in january_healthcare if c.status == "denied"] == ["CL-2048"]


def test_document_names_resolve_to_guidance():
    # Claims say "pathology report"; guidance is keyed "original pathology report".
    assert "specimen details" in DATA.document_guidance("pathology report")
    assert "visit date" in DATA.document_guidance("office note")
    assert DATA.document_guidance("diagnosis report") is None
    assert "replacement copy" in DATA.document_alternative_guidance("diagnosis report")


def test_representative_lookup_is_scoped():
    assert DATA.find_representative("David Chen", "P9").relationship == "son"
    assert DATA.find_representative("David Chen", "P7") is None


# --- State machine -----------------------------------------------------------------


def test_cannot_leave_verify_without_verification():
    state = SessionState()
    with pytest.raises(ValueError):
        state.transition(Phase.RESOLVE_INTENT)


def test_cannot_skip_phases():
    state = SessionState(verified_party_id="P9")
    with pytest.raises(ValueError):
        state.transition(Phase.PROCESS_CASE)


def test_happy_path_transitions():
    state = SessionState(verified_party_id="P9")
    for phase in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS, Phase.ENDED):
        state.transition(phase)
    assert state.is_over


def test_escalation_allowed_before_verification():
    state = SessionState()
    state.escalate("verification_failures")
    assert state.phase == Phase.ESCALATED and state.is_over

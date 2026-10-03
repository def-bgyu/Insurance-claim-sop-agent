"""Parser fallback chain, output sanitizing, regex pre-pass, and PII masking."""

import pytest

from backend.llm.extractor import FIELD_NAMES, coerce_extraction, regex_prepass
from backend.llm.parsing import parse_json_object
from backend.llm.sanitize import clean_reply
from backend.trace import mask_text

# --- Parser fallback chain ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, layer",
    [
        ('{"full_name": "Ava Lopez"}', "json"),
        ('Sure! <json>{"full_name": "Ava Lopez"}</json>', "xml_tag"),
        ('```json\n{"full_name": "Ava Lopez"}\n```', "code_fence"),
        ('Here you go: {"full_name": "Ava Lopez"} hope that helps', "braces"),
        ("{'full_name': 'Ava Lopez', 'off_topic': False,}", "repaired"),
        ("<full_name>Ava Lopez</full_name><off_topic>false</off_topic>", "xml_fields"),
        ('<think>the user said Ava...</think>{"full_name": "Ava Lopez"}', "json"),
    ],
)
def test_parser_layers(raw, layer):
    result = parse_json_object(raw, field_names=FIELD_NAMES)
    assert result.layer == layer
    assert result.data["full_name"] == "Ava Lopez"


def test_parser_gives_up_safely():
    assert parse_json_object("I cannot help with that.", field_names=FIELD_NAMES).layer == "failed"


def test_bad_fields_are_dropped_not_fatal():
    ex = coerce_extraction(
        {"full_name": "Ava Lopez", "intent": "buy_a_boat", "case_month": "January", "emotion": 7}
    )
    assert ex.full_name == "Ava Lopez" and ex.intent is None and ex.case_month == 1


def test_status_synonyms():
    assert coerce_extraction({"case_status": "Rejected"}).case_status == "denied"


# --- Sanitizing ---------------------------------------------------------------------


def test_reasoning_and_template_tokens_are_stripped():
    raw = "<think>internal plan</think>Agent: Thanks for waiting.<|im_end|>"
    assert clean_reply(raw) == "Thanks for waiting."


def test_unclosed_reasoning_is_stripped():
    assert clean_reply("Hello there. <think>and now I will") == "Hello there."


# --- Regex pre-pass -----------------------------------------------------------------


def test_regex_prepass_on_demo_utterance():
    found = regex_prepass(
        "My name is Margaret Chen, policy POL-9921. DOB is 1985-03-15, SSN last four is 4472."
    )
    assert found == {"dob": "1985-03-15", "id_last4": "4472", "policy_number": "POL-9921"}


def test_claim_dates_are_not_mistaken_for_birth_dates():
    assert "dob" not in regex_prepass("my claim from 01/12/2026 was denied, why?")


def test_years_near_words_are_not_ids():
    assert "id_last4" not in regex_prepass("the provider office note from 2026")


def test_keyword_hints_backup():
    from backend.llm.extractor import keyword_hints

    assert keyword_hints("calling about my denied healthcare claim from January") == {
        "case_type": "healthcare", "case_status": "denied", "case_month": "January",
    }
    assert "case_month" not in keyword_hints("born March 15, 1985")
    assert "case_type" not in keyword_hints("I take care of it")  # "care" is not "car"


def test_hints_survive_a_failed_model():
    from backend.llm.extractor import extract
    from backend.llm.provider import LLMResult

    class Broken:
        name = model = "broken"

        def complete(self, *args, **kwargs):
            return LLMResult(text="", model="broken", latency_ms=0, error="down")

    outcome = extract(Broken(), "my denied healthcare claim from January, SSN last 4 is 4472", "VERIFY_ID", None, [])
    ex = outcome.extraction
    assert outcome.parse_layer == "llm_error"
    assert (ex.case_type, ex.case_status, ex.case_month, ex.id_last4) == ("healthcare", "denied", 1, "4472")


# --- Masking ------------------------------------------------------------------------


def test_mask_text():
    masked = mask_text(
        "DOB 1985-03-15, SSN last four is 4472, phone +1 650-521-2836, email margaret@email.com"
    )
    for secret in ("03-15", "4472", "521-2836", "margaret@"):
        assert secret not in masked
    assert "***2" in masked and "2836" in masked and "m***@email.com" in masked

import pytest

from backend.data import get_data
from backend.sop.normalize import names_match, normalize_dob, normalize_phone
from backend.sop.spec import IdentityField as F
from backend.sop.verification import normalize_identity_value, verify_identity

RECORDS = get_data().identity_records()


def provided(**raw: str) -> dict[F, str]:
    """Normalize raw caller input the same way the agent will."""
    out = {}
    for name, value in raw.items():
        field = F(name)
        normalized = normalize_identity_value(field, value)
        if normalized is not None:
            out[field] = normalized
    return out


# --- Normalization -----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["1985-03-15", "03/15/1985", "March 15, 1985", "15 March 1985", "03151985", "19850315",
     "march 15 1985"],
)
def test_dob_formats(raw):
    assert normalize_dob(raw) == "1985-03-15"


def test_incomplete_dob_is_rejected_not_guessed():
    assert normalize_dob("March 1985") is None


def test_phone_formats():
    assert normalize_phone("+1 (650) 521-2836") == "6505212836"
    assert normalize_phone("650.521.2836") == "6505212836"
    assert normalize_phone("521-2836") is None


@pytest.mark.parametrize(
    "given",
    ["Margaret Chen", "margaret chen", "Margaret A. Chen", "Chen Margaret", "MARGARET  CHEN"],
)
def test_name_leniency(given):
    assert names_match(given, ["Margaret Chen"])


@pytest.mark.parametrize("given", ["Maggie Chen", "Margaret", "Margaret Smith"])
def test_name_strictness(given):
    assert not names_match(given, ["Margaret Chen"])


def test_name_alias_and_spacing():
    assert names_match("Yaven Li", ["Ya Wen Li", "Yaven Li"])
    assert names_match("Yawen Li", ["Ya Wen Li"])


# --- 3-of-5 verification -----------------------------------------------------------


def test_demo_caller_is_verified():
    # "My name is Margaret Chen ... DOB is 1985-03-15, SSN last four is 4472."
    result = verify_identity(
        provided(full_name="Margaret Chen", dob="1985-03-15", id_last4="4472"), RECORDS
    )
    assert result.verified and result.party_id == "P9"


def test_two_fields_is_not_enough_and_not_a_failed_attempt():
    result = verify_identity(provided(full_name="Margaret Chen", dob="1985-03-15"), RECORDS)
    assert not result.verified
    assert not result.evaluated  # doesn't count toward the retry limit
    assert result.missing_fields == [F.ID_LAST4, F.PHONE, F.EMAIL]


def test_one_wrong_field_out_of_three_fails():
    result = verify_identity(
        provided(full_name="Margaret Chen", dob="1985-03-15", id_last4="0000"), RECORDS
    )
    assert not result.verified and result.evaluated


def test_wrong_field_recovered_by_a_fourth():
    result = verify_identity(
        provided(
            full_name="Margaret Chen", dob="1985-03-15", id_last4="0000",
            email="margaret@email.com",
        ),
        RECORDS,
    )
    assert result.verified and result.party_id == "P9"


def test_nickname_needs_other_fields():
    assert not verify_identity(
        provided(full_name="Maggie Chen", dob="1985-03-15", id_last4="4472"), RECORDS
    ).verified
    assert verify_identity(
        provided(
            full_name="Maggie Chen", dob="1985-03-15", id_last4="4472", phone="650-521-2836"
        ),
        RECORDS,
    ).verified


def test_fields_from_different_people_do_not_combine():
    # Margaret's name + Ava's DOB + Ma Tian's ID: 3 "real" values, no single record.
    result = verify_identity(
        provided(full_name="Margaret Chen", dob="1990-08-21", id_last4="6688"), RECORDS
    )
    assert not result.verified


def test_near_identical_phone_does_not_match():
    # Ya Wen Li's phone differs from Margaret's by one digit.
    result = verify_identity(
        provided(full_name="Margaret Chen", dob="1985-03-15", phone="+16505212830"), RECORDS
    )
    assert not result.verified


def test_email_alias_counts():
    result = verify_identity(
        provided(full_name="Ya Wen Li", dob="12/03/1989", email="yawen.li@example.com"),
        RECORDS,
    )
    assert result.verified and result.party_id == "P13"


def test_national_id_counts_as_id_last4():
    result = verify_identity(
        provided(full_name="Ma Tian", dob="1964-09-10", id_last4="6688"), RECORDS
    )
    assert result.verified and result.party_id == "P12"


def test_full_ssn_is_rejected():
    assert normalize_identity_value(F.ID_LAST4, "123-45-4472") is None

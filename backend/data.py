#reads data from fixture and works as a data endpoint.

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from backend.config import FIXTURES_DIR


# --- Models ---------------------------------------------------------------------------

class Policyholder(BaseModel):
    party_id: str
    name: str
    policy_number: str
    dob: str  # YYYY-MM-DD
    id_type: str  # "ssn_last4" | "national_id_last4"
    id_last4: str
    phone: str
    email: str
    name_aliases: list[str] = Field(default_factory=list)
    phone_aliases: list[str] = Field(default_factory=list)
    email_aliases: list[str] = Field(default_factory=list)

    @property
    def all_names(self) -> list[str]:
        return [self.name, *self.name_aliases]

    @property
    def all_phones(self) -> list[str]:
        return [self.phone, *self.phone_aliases]

    @property
    def all_emails(self) -> list[str]:
        return [self.email, *self.email_aliases]


class Claim(BaseModel):
    case_id: str
    party_id: str
    case_type: str
    created_at: date
    status: str
    summary: str
    denial_reason: str | None = None
    documents_needed: list[str] = Field(default_factory=list)
    appeal_deadline: date | None = None
    # Amounts are decimal strings in USD (see claim_schema.json).
    expected_reimbursement_amount: str
    allowed_max_amount: str
    net_pay: str
    net_fee: str


class Representative(BaseModel):
    rep_name: str
    relationship: str
    buyer_name: str
    buyer_party_id: str


class FollowupTopic(BaseModel):
    topic: str
    intent_hints: list[str]
    requires_documents: bool
    match_any: list[str] = Field(default_factory=list)
    template: str = Field(alias="en")


# --- Loading --------------------------------------------------------------------------


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class InsuranceData:
    """In-memory view over the fixture directory."""

    def __init__(self, fixtures_dir: Path):
        self._policyholders = [
            Policyholder(**p) for p in _load_json(fixtures_dir / "policyholders.json")
        ]
        self._claims = [Claim(**c) for c in _load_json(fixtures_dir / "claims.json")]
        self._representatives = [
            Representative(**r) for r in _load_json(fixtures_dir / "representatives.json")
        ]
        self._consent_scenarios = _load_json(fixtures_dir / "consent_scenarios.json")
        self._claim_schema = _load_json(fixtures_dir / "claim_schema.json")
        self._guidance = _load_json(fixtures_dir / "required_document_guideline.json")
        self._followup_topics = [
            FollowupTopic(**t) for t in self._guidance["claim_followup_guidance"]
        ]

    # --- Identity (used only by sop/verification.py) ----------------------------------

    def identity_records(self) -> list[Policyholder]:
        return list(self._policyholders)

    def get_policyholder(self, party_id: str) -> Policyholder | None:
        return next((p for p in self._policyholders if p.party_id == party_id), None)

    def find_representative(self, rep_name: str, party_id: str) -> Representative | None:
        """An authorized representative registered for this policyholder, if any."""
        wanted = " ".join(rep_name.lower().split())
        return next(
            (
                r
                for r in self._representatives
                if r.buyer_party_id == party_id and " ".join(r.rep_name.lower().split()) == wanted
            ),
            None,
        )

    def consent_status_sequence(self, scenario: str = "default") -> list[str]:
        return list(self._consent_scenarios[scenario]["status_sequence"])

    # --- Claims (always scoped to a verified party) ------------------------------------

    def list_claims(self, party_id: str) -> list[Claim]:
        return [c for c in self._claims if c.party_id == party_id]

    def get_claim(self, party_id: str, case_id: str) -> Claim | None:
        """Returns None if the claim does not exist OR belongs to someone else."""
        return next(
            (c for c in self._claims if c.party_id == party_id and c.case_id == case_id),
            None,
        )

    def field_descriptions(self) -> dict[str, str]:
        return {
            name: spec["description"]
            for name, spec in self._claim_schema["field_descriptions"].items()
        }

    # --- Guidance (not customer-specific) ----------------------------------------------

    def _resolve_document_key(self, section: str, document: str) -> str | None:
        """Claims say "pathology report"; guidance says "original pathology report".

        Match exactly first, then on word containment in either direction.
        """
        keys = self._guidance[section].keys()
        doc = document.lower().strip()
        if doc in keys:
            return doc
        doc_words = set(doc.split())
        for key in keys:
            key_words = set(key.split())
            if doc_words <= key_words or key_words <= doc_words:
                return key
        return None

    def document_guidance(self, document: str) -> str | None:
        key = self._resolve_document_key("document_guidance", document)
        return self._guidance["document_guidance"][key]["en"] if key else None

    def document_alternative_guidance(self, document: str) -> str:
        key = self._resolve_document_key("document_alternative_guidance", document)
        section = self._guidance["document_alternative_guidance"]
        return section[key]["en"] if key else section["default"]["en"]

    def default_submission_guidance(self) -> str:
        return self._guidance["default_guidance"]["en"]

    def case_type_guidance(self, case_type: str) -> str | None:
        entry = self._guidance["case_type_guidance"].get(case_type)
        return entry["en"] if entry else None

    def followup_topics(self) -> list[FollowupTopic]:
        return list(self._followup_topics)

    def followup_settings(self) -> dict[str, str]:
        return {k: v["en"] for k, v in self._guidance["claim_followup_settings"].items()}

    def followup_fallback(self) -> str:
        return self._guidance["claim_followup_fallback"]["en"]


@lru_cache(maxsize=1)
def get_data() -> InsuranceData:
    return InsuranceData(FIXTURES_DIR)

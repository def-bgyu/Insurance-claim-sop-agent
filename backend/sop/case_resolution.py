"""RESOLVE_INTENT helper: match remembered case hints to the verified caller's claims.

Pure code, no LLM: the LLM only turned "my denied healthcare claim from January"
into hints; here every stated hint must hold for a claim to match.
"""

from backend.data import Claim
from backend.sop.state import CaseHints


def has_hints(hints: CaseHints) -> bool:
    return any([hints.case_id, hints.case_type, hints.status, hints.month, hints.year])


def match_claims(claims: list[Claim], hints: CaseHints) -> list[Claim]:
    if hints.case_id:
        return [c for c in claims if c.case_id == hints.case_id]
    matches = claims
    if hints.case_type:
        matches = [c for c in matches if c.case_type == hints.case_type]
    if hints.status:
        matches = [c for c in matches if c.status == hints.status]
    if hints.month:
        matches = [c for c in matches if c.created_at.month == hints.month]
    if hints.year:
        matches = [c for c in matches if c.created_at.year == hints.year]
    # Newest first, so lists read naturally.
    return sorted(matches, key=lambda c: c.created_at, reverse=True)


def describe_claim(claim: Claim) -> str:
    """Short label used when asking the caller to confirm or choose a claim."""
    article = "an" if claim.case_type[:1] in "aeiou" else "a"
    return (
        f"claim {claim.case_id}, {article} {claim.case_type} claim filed on "
        f"{claim.created_at:%B} {claim.created_at.day}, {claim.created_at.year} "
        f"(status: {claim.status})"
    )

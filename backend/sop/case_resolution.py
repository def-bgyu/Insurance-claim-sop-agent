"""RESOLVE_INTENT helper: match remembered case hints to the verified caller's claims.

Pure code, no LLM: the LLM only turned "my denied healthcare claim from January"
into hints; here every stated hint must hold for a claim to match.
"""

from backend.data import Claim
from backend.sop.state import CaseHints


def has_hints(hints: CaseHints) -> bool:
    return any([hints.case_id, hints.case_type, hints.status, hints.month, hints.day, hints.year])


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
    if hints.day:
        matches = [c for c in matches if c.created_at.day == hints.day]
    if hints.year:
        matches = [c for c in matches if c.created_at.year == hints.year]
    # Newest first, so lists read naturally.
    return sorted(matches, key=lambda c: c.created_at, reverse=True)


def distinguishing_details(claims: list[Claim]) -> list[str]:
    """Which details would tell these claims apart, phrased for the caller.
    Lets the agent narrow down without listing the claims."""
    details = []
    if len({c.case_type for c in claims}) > 1:
        details.append("what type of claim it is")
    if len({c.created_at for c in claims}) > 1:
        details.append("when it was filed")
    if len({c.status for c in claims}) > 1:
        details.append("its status")
    details.append("the claim number")
    return details


def describe_claim(claim: Claim) -> str:
    """Short label used when asking the caller to confirm or choose a claim."""
    article = "an" if claim.case_type[:1] in "aeiou" else "a"
    return (
        f"claim {claim.case_id}, {article} {claim.case_type} claim filed on "
        f"{claim.created_at:%B} {claim.created_at.day}, {claim.created_at.year} "
        f"(status: {claim.status})"
    )

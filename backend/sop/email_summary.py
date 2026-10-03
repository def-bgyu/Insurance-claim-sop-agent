"""POST_PROCESS: build the email summary preview from session state.

Built deterministically from what was actually discussed (state.discussed), so the
email can't contain anything the conversation didn't establish. Sending is
simulated: the assessment's email addresses are dummies.
"""

from pydantic import BaseModel

from backend.data import InsuranceData
from backend.sop.grounding import HUMAN_REVIEW_STEP
from backend.sop.state import SessionState


class EmailPreview(BaseModel):
    to: str
    subject: str
    body: str


def build_email_preview(state: SessionState, data: InsuranceData) -> EmailPreview | None:
    if not state.verified:
        return None
    holder = data.get_policyholder(state.verified_party_id)

    lines = [f"Hello {holder.name},", "", "Here is a summary of your call with claims support.", ""]

    case_ids = list(dict.fromkeys(item.case_id for item in state.discussed))
    lines.append("WHAT WE DISCUSSED")
    if not case_ids:
        lines.append("- General questions about your account. No specific claim was reviewed.")
    for case_id in case_ids:
        items = [i for i in state.discussed if i.case_id == case_id]
        topics = ", ".join(dict.fromkeys(i.intent.value.replace("_", " ") for i in items))
        lines.append(f"- Claim {case_id}: {topics}")

    lines += ["", "CLAIM STATUS / OUTCOME"]
    for case_id in case_ids:
        claim = data.get_claim(state.verified_party_id, case_id)
        lines.append(f"- {case_id} ({claim.case_type}): {claim.status}. {claim.summary}.")
        if claim.denial_reason:
            lines.append(f"  Reason: {claim.denial_reason}.")

    next_steps = list(dict.fromkeys(s for i in state.discussed for s in i.next_steps))
    if state.handoff_requested and HUMAN_REVIEW_STEP not in next_steps:
        next_steps.append("A human claims representative will follow up with you.")
    lines += ["", "NEXT STEPS"]
    lines += [f"- {s}" for s in next_steps] or ["- No further action is needed from you."]

    lines += ["", "Thank you for contacting us.", "Claims Support"]
    return EmailPreview(
        to=holder.email,
        subject="Summary of your claims support conversation",
        body="\n".join(lines),
    )

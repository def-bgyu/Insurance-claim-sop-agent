# The SOP: business rules

The workflow the agent follows. The source of truth in code is
[`backend/sop/spec.py`](../backend/sop/spec.py).

## 1. VERIFY_ID

**Goal:** confirm who the caller is before anything about claims is shared.

- **Rule:** at least **3 of these 5** fields must match **one** policyholder record:
  full legal name · date of birth · phone · email · last 4 of SSN or national ID.
- **Not counted:** the policy number (it identifies the account, so it's used only
  as a hint).
- **Names:** formatting is flexible (case, punctuation, a middle name or initial not
  on file, given/family order, spacing, aliases on file). **Nicknames don't match**
  ("Maggie" ≠ "Margaret"); the caller can give another field instead.
- **Dates and phones** are accepted in any common format. An incomplete date
  ("March 1985") is never guessed.
- **A full SSN is rejected:** only the last four digits are ever needed.
- **Mismatch:** the agent never says *which* field didn't match; it asks for the
  remaining fields instead. "Not enough fields yet" is not a failed attempt.
- **Limit:** 3 failed attempts → transfer to a human.
- **Before verification** no claim details are shared, and the caller isn't
  addressed by name (any name typed is unconfirmed).
- **Remembered for later:** anything said about the claim (type, status, month,
  date, claim number, intent).

## 2. RESOLVE_INTENT

**Goal:** find the one claim the caller means, without listing their claims.

- If the caller already described the claim, the remembered hints are used to
  **confirm it** ("Just to confirm, are you calling about claim CL-2048…?").
- Otherwise: "I see you have some claims with us. Which claim are you calling
  about?" The caller can describe it by claim number, type, status, month, year,
  or exact filing date.
- Several matches → the agent asks for the detail that tells them apart (code works
  out which details actually differ). No list.
- No match → "I couldn't find a … on your account", ask for more detail. No list.
- **Every inferred claim is confirmed**; only an exact claim number skips it.
- No claims on file → "I don't see any existing claims with us. What can I help
  you with today?"

**Intents** (what the caller wants): status inquiry · denial question · payment
question · document submission · next steps · general claim question · speak to a
human.

## 3. PROCESS_CASE

**Goal:** answer naturally, using only grounded data.

- Facts come only from `claims.json`, `claim_schema.json` and
  `required_document_guideline.json`. The agent never invents amounts, dates,
  reasons or procedures.
- The guideline file's prepared answers (how/where/when to submit, processing
  time, file formats, alternatives if a document is missing) are matched by phrase
  first, then by the model's topic choice.
- **Deadline rule:** if a claim's appeal deadline has passed (real current date),
  the claim record wins over the generic "submit within a week" guidance: the
  agent explains the deadline has passed and offers a human representative.
- The caller can switch to another claim at any time.

## 4. POST_PROCESS

**Goal:** offer a summary email; the caller chooses.

- The summary covers what was discussed, the claim status/outcome, and next steps.
  It can be previewed (**Preview email**). Sending is simulated.
- **It only ever goes to the email address on file.** "Send it to a different
  email" is not consent: the agent explains why and offers a human who can update
  the address.

## Across all phases

**Scope.** Off-topic questions are declined politely. The 3rd one → transfer.

**Requests the agent can't fulfil** (filing a new claim, changing the policy,
updating any personal detail) → the agent says so and **offers a human**:
offer → one gentle re-offer if declined → stop and ask what else it can help with.
"Yes" connects immediately, or offers the email first if a claim was discussed.

**The agent never changes personal details.**

**Emotions** (each handled differently, before continuing with the workflow):

| Emotion | Response | Counts toward transfer? |
|---|---|---|
| Frustration | Brief acknowledgment of the situation, keep moving | yes |
| Anger | Brief sincere apology, no blame, calm and short | yes |
| Refusal | Explain why it's needed; any other ID field works instead | yes |
| Anxiety | Reassure: information is protected, we'll work through it | no |
| Confusion | Plain words, one thing at a time | no |

Frustration, anger and refusal together: the 3rd such turn → transfer. The agent
never comments on feelings the caller hasn't expressed.

**Human transfer** whenever the caller asks for one (after the email offer if a
claim was discussed), or when a limit is reached. The same question asked a 3rd
time in a row without progress → offer a human.

## Authorized representatives

From `representatives.json` (e.g. David Chen, son of Margaret Chen):

1. The representative must be **on file** for that policyholder (otherwise →
   transfer), and give **3 of the policyholder's** identity fields.
2. A **consent request** goes to the policyholder's phone. Only the policyholder's
   decision grants access; nothing the caller says does.
3. Approved → the agent helps with the policyholder's claims, addresses the
   representative by name, and emails the summary to the **policyholder's** address.
4. Denied or no response → explained, and a human offered.

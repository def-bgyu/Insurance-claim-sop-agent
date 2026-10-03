# Changelog

What changed, and why. Each entry maps to a git commit, so any change can be
inspected (`git show <commit>`) or rolled back (`git revert <commit>`).

---

## 0.6.0: Personal details are never changed; final messages are exact

**Found in live testing.** At the email step Margaret said "Yes, can you send it to
a different email? The one on file can't be accessed by me." The code read "Yes",
sent the summary to the address on file and transferred her, while Haiku replied
"What's the email address you'd like me to send the summary to instead?", a
question in a chat that had already ended, offering something the system can't do.

**Rule (decided by Nidhi):** the agent is never in a position to update personal
details. Such requests get the human-offer flow, and the summary only ever goes to
the email address on file.

**Changes**
1. **Personal-detail requests** (email, phone, address…) are unsupported requests in
   every phase → offer a human. The extractor prompt and the responder prompt both
   state the agent cannot change personal details.
2. **During wrap-up**, "send it to a different email" (or typing an email that isn't
   on file, a deterministic backup) is **not consent**. The agent explains it can
   only use the address on file, then:
   - transfer already arranged → says the representative can help, re-asks the
     email question;
   - otherwise → offers a human; if declined twice, returns to the email question.
3. **Final messages are the code's exact text** (transfers and goodbyes,
   `use_llm=False`). They must say exactly where the email went and what happens next.
4. **Model-added questions are removed on every turn**, not just turns where code
   appends its own question. (This was deferred in 0.5.0.)

---

## 0.5.0: The agent only uses the verified customer's name

**Found in live testing.** Verified as Ya Wen Li (alias "Yaven Li"), the caller
then pasted Margaret Chen's details. The session correctly stayed on Ya Wen's
account (no data exposed), but the agent started calling the caller "Margaret".
**Cause:** the responder was never told who the verified customer was, so it
took the most recent name from the chat.

**Changes** (agreed with Nidhi: retry once, then fallback)
1. **After verification**, the responder is given the customer's name from the
   record and told to address them only by it (given name, e.g. "Ya Wen").
2. **Before verification**, the agent doesn't address the caller by name at all,
   since any name typed then is unconfirmed.
3. **Name guard** (`responder.name_violations`): every name the caller mentions is
   remembered (`state.names_mentioned`). A reply using one that isn't the verified
   customer's (or any of them, before verification) is **regenerated once** with a
   correction; if the retry is still wrong, the safe fallback (which never contains
   names) is sent. Traces show `retried: true` when this happens.

Not changed (discussed, deferred): the chat can close on a message that asks a
question while the model appended its own question to the goodbye; and a second
identity after verification isn't called out explicitly.

---

## 0.4.0: Claims are never listed; the caller describes the claim

**Feedback from live testing.** Right after verification the agent listed every
claim on the account with IDs, types, dates and statuses. That's more than the
caller needs to see, and more than a support agent should volunteer.

**Changes** (wording agreed with Nidhi)
1. **Claims on file, caller hasn't said which:** "I see you have some claims with
   us. Which claim are you calling about?" No list; the model is given no claim
   facts at all for that turn.
2. **Caller identifies the claim by any claim detail:** claim number, type,
   status, month/year, or the exact filing date. New `case_day` hint so "the one
   filed January 28th" works.
3. **Description matches several claims:** no list. Code works out which details
   actually differ between them (`case_resolution.distinguishing_details`) and asks
   for one: e.g. "I see more than one healthcare claim on your account. To find
   the right one, could you tell me when it was filed, its status, or the claim
   number?"
4. **Nothing matches:** "I couldn't find a … on your account", then asks for more
   detail, still without revealing what the caller does have.
5. **No claims on file:** "Thank you for verifying your identity. I don't see any
   existing claims with us. What can I help you with today?" A question about a
   claim anyway, or a request we can't handle, goes to the human-offer flow;
   "nothing" ends the chat.
6. Only the single claim the caller pointed to is ever named, in the
   confirmation question ("Just to confirm, are you calling about claim CL-2048…?").

Note: the policy number identifies the account, not a claim (all of Margaret's
claims are under POL-9921), so it stays a lookup hint only.

---

## 0.3.0: Human-transfer flow, no-claims callers, action guard, loop safety net

**Bug found in live testing (Ava Lopez, who has no claims).** After verification
the agent said "I can see you have claims on your account" and then asked "Which
claim are you calling about?" every turn, forever. When Ava asked to file a new
claim, Haiku repeatedly said "I'm connecting you to our claims filing team", but
no transfer happened, because only code can transfer.

**Root causes:** (a) no path for a verified caller with zero claims (the code
listed an empty claim list); (b) the responder could describe actions the code
never took; (c) no path for in-scope-but-unsupported requests; (d) nothing
detected a conversation going in circles.

**Changes**
1. **One reusable "offer a human" flow** (designed with Nidhi):
   offer ("I can't help with X here, but a human representative can. Would you
   like me to connect you?") → on "no", **one** gentle re-offer explaining it's the
   recommended route → on a second "no", stop persuading and ask "Is there
   anything else I can help you with?". Insisting "no, I want *you* to do it"
   counts as a "no". Saying "that's all" is respected and never re-offered.
   - On "yes" with **nothing discussed yet** → connect immediately, chat ends.
   - On "yes" **after a claim was discussed** → offer the email summary first,
     then connect (POST_PROCESS requirement). New transition
     `RESOLVE_INTENT → POST_PROCESS` for this.
   - Used by: unsupported requests, no claims on file, appeal deadline passed,
     and the stuck-loop safety net.
2. **No claims on file:** said plainly, offer a human; "no" to "anything else?"
   ends the chat (new transition `RESOLVE_INTENT → ENDED`). No email offered:
   nothing to summarize.
3. **Unsupported requests:** new extractor field `unsupported_request` (e.g.
   "filing a new claim", "changing your policy"). The agent names what it can't do
   and offers a human. A new, different request gets its own offer.
4. **Action guard** (`responder.action_violations`): if the reply claims a
   transfer/connection/email ("I'm connecting you", "I've sent…") that the code's
   own reply for this turn doesn't contain, the safe fallback is sent instead.
   The prompt also says this explicitly.
5. **Stuck-loop safety net:** the same closing question a 3rd time in a row →
   offer a human instead. (Human and email offers are exempt; they have their
   own limits.)

Tests: regression tests replay the Ava conversation; plus both "yes" paths, the
two-decline flow, the action guard, and the loop detector (83 total).

---

## 0.2.0: Yes/no answers are always matched to the question actually asked

**Bug found in live testing (Haiku 4.5, trace `9c405e18…`).** The caller picked a
claim by saying "January 22th". The code silently selected CL-2048 (the only
January claim) and planned an answer ending with "Would you like a human
representative?", but Haiku instead asked "is this the right claim?" (it noticed
the 22nd vs. 12th mismatch). The caller's "yes that one" was then read as
accepting a human transfer, and "Why do I have to talk to a human here?" was
misread as a request for one, which escalated the call.

**Root cause:** the code didn't control which question was on screen, so the
state and the caller could disagree about what a "yes" meant.

**Changes**
1. **Code writes the closing question** (`ResponsePlan.closing_question`). The
   LLM writes only the body; the responder strips any trailing question the model
   adds and appends the code's question word for word. *Why:* what the caller is
   asked always matches what the state expects.
2. **One `pending_question` in state** (`confirm_claim`, `choose_claim`,
   `anything_else`, `offer_human`, `offer_email`) replaces the separate
   `awaiting_confirmation` / `human_offered` flags. Every yes/no is interpreted
   against it. A "yes" with hints that merely restate the candidate claim still
   confirms; only contradicting hints block it.
3. **Every inferred claim is confirmed**, including picks from a list ("the
   January one", "the auto one"). Only an exact claim ID skips confirmation.
   *Why:* would have caught the 22nd vs. 12th mismatch in code. Costs one turn.
4. **`wants_human` only for an actual request.** The extractor prompt now says a
   question or complaint about a transfer is not a request. During wrap-up, a
   human request never skips the email question; the agent can explain why a human
   is needed using the facts already established (new `_wrap_up_facts`).

Also: "no" to "anything else?" now ends the case (previously only "that's all"
did), and declining the human offer continues normally.

Tests: 5 regression tests replay the failing conversation and its variations.

---

## 0.1.0: Initial SOP harness (baseline)

The first working version. Key design decisions, all made before writing code:

**Architecture: code owns the workflow, the LLM owns the conversation**
- Phases (`VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS`) live in
  server-side session state. Only code changes phase (`SessionState.transition`),
  and it refuses to leave `VERIFY_ID` unless the caller is verified.
- Each turn: extract (LLM + regex) → remember → guards → phase logic → respond → trace.
- The LLM never calls tools. It classifies; code calls the data layer. *Why:* small
  models are weakest at tool calling, and this keeps the harness model-agnostic.

**Identity verification**
- At least 3 of 5 fields (full name, DOB, phone, email, SSN/national ID last 4) must
  match ONE record. The policy number is a lookup hint only and never counts.
- Mismatch messages never say which field failed; they ask for the remaining fields.
- Names: lenient on formatting (case, punctuation, middle names, order, spacing,
  fixture aliases), strict on identity: nicknames do not match. The opening
  greeting asks for the "full legal name" instead of hinting after a failure.
- "Not enough fields yet" is not a failed attempt; only real mismatches count.

**Grounding**
- Claim facts are built by code from the fixtures (`grounding.py`). The responder
  can rephrase but a guard rejects any claim ID, amount or date not in the facts,
  sending a deterministic fallback reply instead.
- Only the fixtures' own guidance is used; we wrote no guidance of our own.
- Dates use the real current date. If a claim's appeal deadline has passed, the
  claim record wins over the generic "submit within a week" template and the agent
  offers a human representative.

**Limits and escalation**
- 3 off-topic questions, 3 frustrated/angry/refusing turns, or 3 failed
  verifications → transfer to a human. An explicit request for a human is honored.

**Memory across phases**
- Case hints (type, status, month, year, claim ID) and intent are stored whenever
  the caller mentions them, even during `VERIFY_ID`, and used after verification.

**Email (POST_PROCESS)**
- Preview built from what was discussed; sending is simulated (dummy addresses).

**Robustness for any model**
- Parser fallback chain (JSON → XML tag → code fence → braces → repaired → per-field
  tags), reasoning-tag stripping, flat extraction schema, regex + keyword backups.

**Representatives (open item)**
- Authorized representatives (`representatives.json`) are currently always
  transferred to a human; the consent flow (`consent_scenarios.json`) is not yet
  designed. See `TODO(consent)` in `backend/sop/engine.py`.

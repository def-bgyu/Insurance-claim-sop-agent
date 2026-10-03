# Changelog

What changed, and why. Each entry maps to a git commit, so any change can be
inspected (`git show <commit>`) or rolled back (`git revert <commit>`).

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

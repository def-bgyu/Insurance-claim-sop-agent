# Changelog

What changed, and why. Each entry maps to a git commit, so any change can be
inspected (`git show <commit>`) or rolled back (`git revert <commit>`).

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

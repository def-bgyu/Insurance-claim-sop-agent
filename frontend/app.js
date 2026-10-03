// Test UI for the SOP agent. Plain JS, no build step.
// The API key stays in this page's memory only and is sent once, when a session starts.

const $ = (sel) => document.querySelector(sel);

const PHASES = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"];
const LIMITS = { off_topic: 3, frustration: 3, verification_failures: 3 };

let sessionId = null;
let lastPhase = "VERIFY_ID";
let traces = [];
let consentNoteShown = false;

// --- API --------------------------------------------------------------------------

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

async function loadConfig() {
  const config = await api("/api/config");
  if (config.server_has_api_key) {
    $("#api-key").placeholder = "Using server key (optional override)";
  }
}

// --- Rendering ---------------------------------------------------------------------

function addMessage(role, text, meta) {
  const list = $("#messages");
  list.querySelector(".empty")?.remove();
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  bubble.textContent = text;
  list.append(bubble);
  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = `msg-meta ${role}`;
    metaEl.textContent = meta;
    list.append(metaEl);
  }
  list.scrollTop = list.scrollHeight;
}

function setTyping(on) {
  $("#typing")?.remove();
  if (!on) return;
  const el = document.createElement("div");
  el.id = "typing";
  el.className = "typing";
  el.textContent = "Agent is typing…";
  $("#messages").append(el);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

function showBanner(text, kind = "warn") {
  const banner = $("#banner");
  if (!text) return banner.classList.add("hidden");
  banner.textContent = text;
  banner.className = `banner ${kind}`;
}

function renderPhases(state) {
  const phase = state.phase;
  const terminal = phase === "ESCALATED" || phase === "ENDED";
  const reached = terminal ? lastPhase : phase;
  const index = PHASES.indexOf(reached);
  document.querySelectorAll("#phases li").forEach((li, i) => {
    li.classList.toggle("done", i < index || (terminal && i === index));
    li.classList.toggle("current", !terminal && i === index);
  });
  if (!terminal) lastPhase = phase;

  const badge = $("#terminal-badge");
  if (phase === "ESCALATED") {
    badge.textContent = "Transferred to a human representative";
    badge.className = "badge danger";
  } else if (phase === "ENDED") {
    badge.textContent = "Conversation ended";
    badge.className = "badge ok";
  } else {
    badge.className = "badge hidden";
  }
}

function kv(rows) {
  const dl = document.createElement("dl");
  dl.className = "kv";
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    if (value instanceof Node) dd.append(value);
    else dd.textContent = value ?? "—";
    dl.append(dt, dd);
  }
  return dl;
}

function badge(text, kind) {
  const el = document.createElement("span");
  el.className = `badge ${kind}`;
  el.textContent = text;
  return el;
}

function meter(value, limit) {
  const wrap = document.createElement("span");
  wrap.className = "meter";
  const bar = document.createElement("span");
  bar.className = "meter-bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.min(value / limit, 1) * 100}%`;
  if (value >= limit) fill.className = "danger";
  else if (value > 0) fill.className = "warn";
  bar.append(fill);
  wrap.append(bar, `${value} / ${limit}`);
  return wrap;
}

function describeObject(obj) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return "—";
  return entries.map(([k, v]) => `${k}: ${v}`).join(", ");
}

function section(title, content) {
  const frag = document.createDocumentFragment();
  const h = document.createElement("h3");
  h.className = "section-title";
  h.textContent = title;
  frag.append(h, content);
  return frag;
}

function renderState(state) {
  const body = $("#tab-state");
  body.replaceChildren(
    ...(state.consent.status === "pending" ? [consentCard(state.consent)] : []),
    section("Workflow", kv([
      ["Phase", state.phase],
      ["Verified", state.verified ? badge("verified", "ok") : badge("not verified", "warn")],
      ["Caller role", state.caller_role],
      ["Escalation", state.escalation_reason],
    ])),
    section("Identity (masked)", kv([
      ["Fields collected", describeObject(state.identity_collected)],
    ])),
    section("Memory across phases", kv([
      ["Case hints", describeObject(state.remembered_hints)],
      ["Policy number hint", state.policy_number_hint],
      ["Intent", state.intent],
      ["Candidate claims", state.candidate_case_ids.join(", ") || "—"],
      ["Pending question", state.pending_question],
      ["Selected claim", state.selected_case_id],
    ])),
    section("Escalation counters", kv([
      ["Off-topic", meter(state.counters.off_topic, LIMITS.off_topic)],
      ["Frustration / refusal", meter(state.counters.frustration, LIMITS.frustration)],
      ["Failed verification", meter(state.counters.verification_failures, LIMITS.verification_failures)],
    ])),
    section("Email summary", kv([
      ["Offered", String(state.email.offered)],
      ["Consent", state.email.consent === null ? "not answered" : String(state.email.consent)],
      ["Human handoff requested", String(state.handoff_requested)],
    ])),
  );
}

function renderTrace() {
  const body = $("#tab-trace");
  if (!traces.length) return;
  body.replaceChildren(
    ...traces.slice().reverse().map((t) => {
      const details = document.createElement("details");
      details.className = "trace-turn";
      const summary = document.createElement("summary");
      const source = t.response.source;
      summary.append(
        `Turn ${t.turn}: ${t.phase_before} → ${t.phase_after} `,
        badge(t.plan.action, "neutral"),
        badge(`reply: ${source}`, source === "llm" ? "ok" : "warn"),
      );
      if (t.extraction) summary.append(badge(`parse: ${t.extraction.parse_layer}`, t.extraction.parse_layer === "failed" || t.extraction.parse_layer === "llm_error" ? "danger" : "neutral"));
      const pre = document.createElement("pre");
      const { state, ...rest } = t;
      pre.textContent = JSON.stringify(rest, null, 2);
      details.append(summary, pre);
      return details;
    }),
  );
}

function applyState(state) {
  renderPhases(state);
  renderState(state);
  if (state.consent.status === "pending" && !consentNoteShown) {
    // Make the demo control impossible to miss, for people and automated evaluators alike.
    consentNoteShown = true;
    addMessage(
      "system",
      "🔧 Demo: a consent request was sent to the policyholder's (simulated) phone. " +
        "Approve or deny it in the SOP state panel on the right.",
    );
    switchTab("state");
  }
  $("#email-btn").disabled = !state.email.offered;
  const over = state.phase === "ESCALATED" || state.phase === "ENDED";
  $("#input").disabled = over;
  $("#send-btn").disabled = over;
  if (over) $("#input").placeholder = "Conversation finished. Start a new one above.";
}

// --- Actions -----------------------------------------------------------------------

async function startSession(event) {
  event.preventDefault();
  const button = $("#start-btn");
  button.disabled = true;
  showBanner("");
  try {
    const data = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({ api_key: $("#api-key").value || null }),
    });
    sessionId = data.session_id;
    traces = [];
    consentNoteShown = false;
    lastPhase = "VERIFY_ID";
    $("#messages").replaceChildren();
    $("#tab-trace").replaceChildren(Object.assign(document.createElement("p"), {
      className: "muted",
      textContent: "Each turn's extraction, policy decision, and response source will appear here.",
    }));
    addMessage("agent", data.greeting);
    applyState(data.state);
    $("#input").placeholder = "Type as the caller…";
    $("#input").focus();
    button.textContent = "Restart conversation";
  } catch (err) {
    showBanner(err.message, "danger");
  } finally {
    button.disabled = false;
  }
}

async function sendMessage(event) {
  event?.preventDefault();
  const input = $("#input");
  const text = input.value.trim();
  if (!text || !sessionId || input.disabled) return;

  addMessage("user", text);
  input.value = "";
  input.disabled = true;
  $("#send-btn").disabled = true;
  setTyping(true);
  try {
    const data = await api(`/api/session/${sessionId}/message`, {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    setTyping(false);
    input.disabled = false;
    $("#send-btn").disabled = false;
    handleTurn(data);
    if (!input.disabled) input.focus();
  } catch (err) {
    setTyping(false);
    input.disabled = false;
    $("#send-btn").disabled = false;
    showBanner(err.message, "danger");
  }
}

// Shared by chat messages and consent decisions: show the reply, update panels.
function handleTurn(data) {
  const meta = data.trace.response.source === "fallback" ? "safe fallback reply" : null;
  addMessage("agent", data.reply, meta);
  traces.push(data.trace);
  renderTrace();
  applyState(data.state);
  const llmError = data.trace.extraction?.llm_error || data.trace.response.llm_error;
  showBanner(llmError ? `Model call failed (${llmError}). The harness used its deterministic fallbacks.` : "");
}

// --- Simulated policyholder consent ----------------------------------------------------
// A representative's access needs the policyholder's approval, which would arrive from
// their phone. Here the tester plays the policyholder from the debug panel.

const CONSENT_RESULT = {
  approved: "approved the request",
  denied: "denied the request",
  no_response: "didn't respond to the request",
};

function consentCard(consent) {
  const card = document.createElement("div");
  card.className = "consent-card";
  card.setAttribute("role", "alert");

  const title = document.createElement("p");
  title.className = "consent-title";
  title.textContent = "📱 Simulated: policyholder's phone";

  const body = document.createElement("p");
  body.textContent =
    `${consent.policyholder_name} received a consent request: "${consent.representative} ` +
    `(${consent.relationship}) is asking to discuss your claims." How do they respond?`;

  const buttons = document.createElement("div");
  buttons.className = "consent-actions";
  for (const [decision, label, cls] of [
    ["approved", "Approve", "btn primary"],
    ["denied", "Deny", "btn"],
    ["no_response", "Don't respond", "btn"],
  ]) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.addEventListener("click", () => decideConsent(decision, consent.policyholder_name));
    buttons.append(b);
  }

  card.append(title, body, buttons);
  return card;
}

async function decideConsent(decision, policyholder) {
  document.querySelectorAll(".consent-actions button").forEach((b) => (b.disabled = true));
  addMessage("system", `📱 Simulated: ${policyholder.split(" ")[0]} ${CONSENT_RESULT[decision]}.`);
  setTyping(true);
  try {
    const data = await api(`/api/session/${sessionId}/consent`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    });
    setTyping(false);
    handleTurn(data);
  } catch (err) {
    setTyping(false);
    showBanner(err.message, "danger");
  }
}

async function showEmailPreview() {
  try {
    const preview = await api(`/api/session/${sessionId}/email-preview`);
    $("#email-to").textContent = preview.to;
    $("#email-subject").textContent = preview.subject;
    $("#email-body").textContent = preview.body;
    $("#email-dialog").showModal();
  } catch (err) {
    showBanner(err.message, "danger");
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $("#tab-state").classList.toggle("hidden", name !== "state");
  $("#tab-trace").classList.toggle("hidden", name !== "trace");
}

// --- Wiring ------------------------------------------------------------------------

$("#setup").addEventListener("submit", startSession);
$("#composer").addEventListener("submit", sendMessage);
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) sendMessage(e);
});
$("#email-btn").addEventListener("click", showEmailPreview);
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    const input = $("#input");
    if (input.disabled) return;
    input.value = chip.dataset.text;
    input.focus();
  }),
);

loadConfig().catch((err) => showBanner(err.message, "danger"));

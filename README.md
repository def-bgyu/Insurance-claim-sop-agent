

# Insurance Claims SOP Agent

## The problem

An insurance claims support agent has to follow a fixed Standard Operating
Procedure:

```
VERIFY_ID  →  RESOLVE_INTENT  →  PROCESS_CASE  →  POST_PROCESS
```

…while still talking like a person. The verification step must be **strict**: no claim details
until the caller's identity is verified (3 of 5 identity fields), no skipped steps,
no invented facts. Others need **flexible reasoning**: messy language, partial
answers, follow-up questions, frustrated or anxious callers. Anything the caller
says early ("I'm calling about my denied healthcare claim from January") has to be
remembered for later, and off-topic questions politely declined.

**Approach:**
Deterministic code decides the phase, verification, which claim facts may be shared, and every
question the caller must answer. The LLM (Claude Haiku 4.5) understands the caller
and phrases replies, and every reply is checked by code before it's sent.

## Demo

▶️ **[Watch the 1-minute demo](docs/media/sop-agent-demo.mp4)**

Try it live: **https://insurance-claim-sop-agent.onrender.com** (enter your own Anthropic
API key in the header; on the free tier the first load can take up to a minute).

## Run it

You need your own [Anthropic API key](https://console.anthropic.com/). No key is
bundled with this project or the hosted demo. There are two ways to provide it:

1. **In the UI:** paste it into the **Anthropic API key** field in the page
   header, then click **Start conversation**. This is the only option on the
   hosted demo.
2. **When starting the server:** set `ANTHROPIC_API_KEY` (shown below). The UI then
   uses it automatically, and the field can stay empty.

The key is only used for your session's model calls.

If a model call fails (an invalid key, a network error, a malformed reply), the
agent falls back on the deterministic workflow implementation. Verification and the workflow are code, so they keep working, and every turn has a pre-written safe reply to fall back on though the quality of responses might drop.

**Docker**

```bash
docker build -t insurance-sop-agent .
docker run -p 8000:8000 insurance-sop-agent                                  # key entered in the UI
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key insurance-sop-agent    # or key at startup
```

**Python (3.11+)**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn backend.main:app --port 8000                               # key entered in the UI
ANTHROPIC_API_KEY=your-key .venv/bin/uvicorn backend.main:app --port 8000    # or key at startup
```

On Windows, use `.venv\Scripts\` in place of `.venv/bin/`. To set the key in
PowerShell, run `$env:ANTHROPIC_API_KEY = "your-key"` before starting the server.

Then open http://localhost:8000.

**Tests**

```bash
.venv/bin/python -m pytest -q
```

## Deploy (Render)

1. On [render.com](https://render.com), choose **New → Web Service** and connect
   this GitHub repository. Render detects the `Dockerfile`.
2. Pick the **Free** instance type and click **Deploy**. Render sets `PORT`, which
   the container uses automatically. No API key is configured on the server, so
   visitors enter their own in the UI.

Notes: free instances sleep when idle, so the first visit can take up to a minute.
Conversations are kept in memory, so a restart or redeploy resets the in-memory.

## Notes & assumptions

- **Policyholder consent is simulated.** When an authorized representative calls
  (e.g. David Chen for Margaret Chen), the policyholder must approve the request
  from their phone, which can't be done in a demo. Instead, the operator gets a
  pop-up in the inspector panel to **Approve**, **Deny**, or **Don't respond** on
  the policyholder's behalf, so you can see how the agent reacts in each case.
  Nothing the caller types can grant access.
- **The automated tests are written against the provided fixture data**
  (`apps/insurance_claims/fixtures/`). They check specific people and claims, such
  as Margaret Chen and CL-2048, so they will fail if the fixtures are replaced. The
  app itself reads whatever data is in those files.


## More

- [docs/SOP.md](docs/SOP.md): the rules the agent follows
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how it works
- [docs/TESTING.md](docs/TESTING.md): test data and scenarios

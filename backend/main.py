"""FastAPI app: session API for the test UI, and serves the UI itself.

Sessions live in memory on the server. The browser only gets replies and masked
state snapshots; the API key a tester enters is kept with the session's provider
and never echoed back.
"""

import os
import threading
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import ROOT_DIR
from backend.data import get_data
from backend.llm.provider import DEFAULT_ANTHROPIC_MODEL, AnthropicProvider
from backend.sop.email_summary import EmailPreview, build_email_preview
from backend.sop.engine import SOPEngine
from backend.sop.state import SessionState
from backend.trace import public_snapshot

FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_CHOICES = ["claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5"]

app = FastAPI(title="Insurance Claims SOP Agent")


@dataclass
class Session:
    state: SessionState
    engine: SOPEngine
    traces: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


SESSIONS: dict[str, Session] = {}


class NewSessionRequest(BaseModel):
    api_key: str | None = Field(default=None, description="Falls back to ANTHROPIC_API_KEY")
    model: str = DEFAULT_ANTHROPIC_MODEL


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _get(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Start a new conversation.")
    return session


@app.get("/api/config")
def get_config():
    return {
        "server_has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "default_model": DEFAULT_ANTHROPIC_MODEL,
        "models": MODEL_CHOICES,
    }


@app.post("/api/session")
def new_session(req: NewSessionRequest):
    api_key = (req.api_key or "").strip() or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(400, "No API key: enter one in the UI or set ANTHROPIC_API_KEY.")
    if req.model not in MODEL_CHOICES:
        raise HTTPException(400, f"Unsupported model. Choose one of {MODEL_CHOICES}.")

    engine = SOPEngine(AnthropicProvider(api_key=api_key, model=req.model), get_data())
    state = engine.start()
    SESSIONS[state.session_id] = Session(state, engine)
    return {
        "session_id": state.session_id,
        "greeting": state.transcript[0].text,
        "state": public_snapshot(state),
    }


@app.post("/api/session/{session_id}/message")
def send_message(session_id: str, req: MessageRequest):
    session = _get(session_id)
    with session.lock:
        result = session.engine.handle(session.state, req.text.strip())
        session.traces.append(result.trace)
        return {"reply": result.reply, "state": public_snapshot(session.state), "trace": result.trace}


@app.get("/api/session/{session_id}/trace")
def get_trace(session_id: str):
    return _get(session_id).traces


@app.get("/api/session/{session_id}/email-preview", response_model=EmailPreview)
def email_preview(session_id: str):
    session = _get(session_id)
    if not session.state.email.offered:
        raise HTTPException(409, "The email summary hasn't been offered yet.")
    preview = build_email_preview(session.state, session.engine.data)
    if preview is None:
        raise HTTPException(409, "No verified caller for this session.")
    return preview


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")

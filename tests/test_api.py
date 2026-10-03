import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.trace import TraceWriter
from tests.fakes import ScriptedProvider
from tests.test_engine import MARGARET_EXTRACTION, MARGARET_OPENING


@pytest.fixture
def client(monkeypatch):
    script = [MARGARET_EXTRACTION, {"confirms_case": "yes"}, {"wants_to_end": True}, {"email_consent": "no"}]
    monkeypatch.setattr(main, "AnthropicProvider", lambda **_: ScriptedProvider(script))
    monkeypatch.setattr("backend.sop.engine.TraceWriter", lambda: TraceWriter(enabled=False))
    return TestClient(main.app)


def test_requires_an_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.post("/api/session", json={}).status_code == 400


def test_full_conversation_over_http(client):
    session = client.post("/api/session", json={"api_key": "test-key"}).json()
    sid = session["session_id"]
    assert session["state"]["phase"] == "VERIFY_ID"
    assert "api_key" not in str(session)

    # Email preview isn't available before it is offered.
    assert client.get(f"/api/session/{sid}/email-preview").status_code == 409

    for text in (MARGARET_OPENING, "yes", "that's all"):
        r = client.post(f"/api/session/{sid}/message", json={"text": text})
        assert r.status_code == 200

    state = r.json()["state"]
    assert state["phase"] == "POST_PROCESS" and state["email"]["offered"]
    assert state["identity_collected"]["id_last4"] == "***2"  # masked in the snapshot

    preview = client.get(f"/api/session/{sid}/email-preview").json()
    assert preview["to"] == "margaret@email.com" and "CL-2048" in preview["body"]

    final = client.post(f"/api/session/{sid}/message", json={"text": "no thanks"}).json()
    assert final["state"]["phase"] == "ENDED"
    assert len(client.get(f"/api/session/{sid}/trace").json()) == 4


def test_consent_endpoint_and_action_required(monkeypatch):
    from tests.test_engine import DAVID, DAVID_TEXT

    monkeypatch.setattr(main, "AnthropicProvider", lambda **_: ScriptedProvider([DAVID]))
    monkeypatch.setattr("backend.sop.engine.TraceWriter", lambda: TraceWriter(enabled=False))
    client = TestClient(main.app)
    sid = client.post("/api/session", json={"api_key": "test-key"}).json()["session_id"]

    body = client.post(f"/api/session/{sid}/message", json={"text": DAVID_TEXT}).json()
    assert body["action_required"]["type"] == "policyholder_consent"
    assert body["state"]["consent"]["status"] == "pending"

    approved = client.post(f"/api/session/{sid}/consent", json={"decision": "approved"}).json()
    assert approved["state"]["verified"] and "action_required" not in approved
    # A second decision is refused: consent is no longer pending.
    assert client.post(f"/api/session/{sid}/consent", json={"decision": "denied"}).status_code == 409


def test_unknown_session_is_404(client):
    assert client.post("/api/session/nope/message", json={"text": "hi"}).status_code == 404


def test_ui_is_served(client):
    assert "Claims SOP Agent" in client.get("/").text

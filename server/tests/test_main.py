import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

SSE_BODY = (
    b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def llama():
    with respx.mock(base_url=settings.llama_server_url, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def qdrant():
    with respx.mock(base_url=settings.qdrant_url, assert_all_called=False) as mock:
        yield mock


def test_health_reports_llm_and_qdrant(client, llama, qdrant):
    llama.get("/health").respond(200, json={"status": "ok"})
    qdrant.get("/collections").respond(
        200, json={"result": {"collections": []}, "status": "ok", "time": 0.0}
    )

    res = client.get("/health")

    assert res.status_code == 200
    assert res.json() == {"status": "ok", "llm": "ok", "db": "ok"}


def test_health_reports_qdrant_down(client, llama, qdrant):
    llama.get("/health").respond(200, json={"status": "ok"})
    qdrant.get("/collections").mock(side_effect=httpx.ConnectError("refused"))

    res = client.get("/health")

    assert res.json()["db"].startswith("error: ")


def test_chat_streams_llama_response_through(client, llama):
    route = llama.post("/v1/chat/completions").respond(
        200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
    )
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}

    res = client.post("/v1/chat/completions", json=payload)

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.content == SSE_BODY
    assert route.calls.last.request.read() == httpx.Request("POST", "/", json=payload).read()


def test_chat_passes_through_llama_errors(client, llama):
    llama.post("/v1/chat/completions").respond(400, json={"error": {"message": "bad request"}})

    res = client.post("/v1/chat/completions", json={"messages": []})

    assert res.status_code == 400
    assert res.json() == {"error": {"message": "bad request"}}


def test_chat_returns_502_when_llama_is_down(client, llama):
    llama.post("/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))

    res = client.post("/v1/chat/completions", json={"messages": []})

    assert res.status_code == 502
    assert "Cannot reach llama-server" in res.json()["detail"]

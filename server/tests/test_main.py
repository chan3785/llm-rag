import base64
import io
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app

SSE_BODY = (
    b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
    b"data: [DONE]\n\n"
)




class FakeClip:
    def __init__(self, model_name: str):
        self.images = []

    def embed_images(self, images):
        self.images.extend(images)
        return [[0.1, 0.2, 0.3, 0.4] for _ in images]


def image_data_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "orange").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_message(text: str = "what is this?") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url()}},
            {"type": "text", "text": text},
        ],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.ClipEmbedder", FakeClip)
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


QUERY_PATH = f"/collections/{settings.qdrant_collection}/points/query"


def sent_messages(route) -> list[dict]:
    return json.loads(route.calls.last.request.read())["messages"]


def test_chat_with_image_prepends_retrieved_context(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant_route = qdrant.post(QUERY_PATH).respond(
        200,
        json={
            "result": {
                "points": [
                    {"id": 1, "version": 0, "score": 0.93, "payload": {"description": "orange cat"}},
                    {"id": 2, "version": 0, "score": 0.85, "payload": {"description": ""}},
                ]
            },
            "status": "ok",
            "time": 0.0,
        },
    )

    res = client.post("/v1/chat/completions", json={"messages": [image_message()], "stream": True})

    assert res.status_code == 200
    query = json.loads(qdrant_route.calls.last.request.read())
    assert query["limit"] == settings.rag_top_k
    assert query["score_threshold"] == settings.rag_score_threshold
    assert len(client.app.state.clip.images) == 1

    messages = sent_messages(llm_route)
    assert messages[0]["role"] == "system"
    assert "- orange cat (similarity 0.93)" in messages[0]["content"]
    assert "similarity 0.85" not in messages[0]["content"]
    assert messages[1] == image_message()


def test_chat_with_image_skips_context_when_nothing_matches(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant.post(QUERY_PATH).respond(
        200, json={"result": {"points": []}, "status": "ok", "time": 0.0}
    )

    client.post("/v1/chat/completions", json={"messages": [image_message()]})

    assert sent_messages(llm_route) == [image_message()]


def test_chat_with_image_still_works_when_retrieval_fails(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant.post(QUERY_PATH).respond(404, json={"status": {"error": "Collection not found"}})

    res = client.post("/v1/chat/completions", json={"messages": [image_message()]})

    assert res.status_code == 200
    assert sent_messages(llm_route) == [image_message()]


def test_chat_without_image_skips_retrieval(client, llama, qdrant):
    llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant_route = qdrant.post(QUERY_PATH)

    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert not qdrant_route.called
    assert client.app.state.clip.images == []

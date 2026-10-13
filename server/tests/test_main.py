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
    sent = json.loads(route.calls.last.request.read())
    assert sent == {**settings.generation_defaults(), **payload}


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


def answer_source(res) -> dict:
    return json.loads(res.headers["x-answer-source"])


def qdrant_points(*points: tuple[float, str]) -> dict:
    return {
        "result": {
            "points": [
                {
                    "id": i,
                    "version": 0,
                    "score": score,
                    "payload": {"landmark": label, "description": f"{label} info"},
                }
                for i, (score, label) in enumerate(points)
            ]
        },
        "status": "ok",
        "time": 0.0,
    }


def test_chat_with_image_prepends_landmark_classification(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant_route = qdrant.post(QUERY_PATH).respond(
        200, json=qdrant_points((0.93, "경복궁"), (0.90, "경복궁"), (0.88, "불국사"))
    )

    res = client.post("/v1/chat/completions", json={"messages": [image_message()], "stream": True})

    assert res.status_code == 200
    assert json.loads(qdrant_route.calls.last.request.read())["limit"] == settings.classify_top_k
    assert len(client.app.state.clip.images) == 1

    messages = sent_messages(llm_route)
    assert messages[0]["role"] == "system"
    assert "'경복궁'(으)로 분류" in messages[0]["content"]
    assert "유사도 0.93, 비슷한 사진 3장 중 2장" in messages[0]["content"]
    assert "명소 정보: 경복궁 info" in messages[0]["content"]
    assert "모른다고 답하세요" in messages[0]["content"]
    assert messages[1] == image_message()

    assert answer_source(res) == {
        "stage": "rag",
        "reason": "matched",
        "landmark": "경복궁",
        "candidate": "경복궁",
        "score": 0.93,
        "threshold": settings.classify_threshold,
        "votes": 2,
        "total": 3,
    }


def test_chat_with_image_warns_llm_when_landmark_is_unknown(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    low = settings.classify_threshold - 0.1
    qdrant.post(QUERY_PATH).respond(200, json=qdrant_points((low, "경복궁")))

    res = client.post("/v1/chat/completions", json={"messages": [image_message()]})

    source = answer_source(res)
    assert (source["stage"], source["reason"]) == ("rag", "below_threshold")
    assert (source["landmark"], source["candidate"]) == (None, "경복궁")
    system = sent_messages(llm_route)[0]
    assert system["role"] == "system"
    assert "등록된 명소 검색(RAG)에서 일치하는 명소를 찾지 못했습니다." in system["content"]
    assert "모르면 모른다고 답하세요" in system["content"]
    assert "경복궁" not in system["content"]


def test_chat_with_image_skips_context_when_nothing_matches(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant.post(QUERY_PATH).respond(
        200, json={"result": {"points": []}, "status": "ok", "time": 0.0}
    )

    res = client.post("/v1/chat/completions", json={"messages": [image_message()]})

    assert sent_messages(llm_route) == [image_message()]
    assert answer_source(res) == {"stage": "vlm", "reason": "no_reference_data"}


def test_chat_with_image_still_works_when_retrieval_fails(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant.post(QUERY_PATH).respond(404, json={"status": {"error": "Collection not found"}})

    res = client.post("/v1/chat/completions", json={"messages": [image_message()]})

    assert res.status_code == 200
    assert sent_messages(llm_route) == [image_message()]
    assert answer_source(res) == {"stage": "vlm", "reason": "retrieval_failed"}


def test_chat_without_image_skips_retrieval(client, llama, qdrant):
    llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant_route = qdrant.post(QUERY_PATH)

    res = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert not qdrant_route.called
    assert client.app.state.clip.images == []
    assert answer_source(res) == {"stage": "vlm", "reason": "no_image"}


def test_chat_sends_only_the_latest_image_to_the_vlm(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)
    qdrant.post(QUERY_PATH).respond(200, json=qdrant_points())
    history = [
        image_message("first"),
        {"role": "assistant", "content": "answer"},
        image_message("second"),
    ]

    client.post("/v1/chat/completions", json={"messages": history})

    first, answer, second = sent_messages(llm_route)
    assert first["content"] == [
        {"type": "text", "text": "(이전에 올린 이미지)"},
        {"type": "text", "text": "first"},
    ]
    assert answer == {"role": "assistant", "content": "answer"}
    assert second == image_message("second")


def test_chat_keeps_client_generation_settings(client, llama, qdrant):
    llm_route = llama.post("/v1/chat/completions").respond(200, content=SSE_BODY)

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "temperature": 0.1,
        },
    )

    sent = json.loads(llm_route.calls.last.request.read())
    assert (sent["max_tokens"], sent["temperature"]) == (50, 0.1)
    assert sent["presence_penalty"] == settings.llm_presence_penalty

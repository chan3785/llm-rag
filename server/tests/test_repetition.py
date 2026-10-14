import asyncio
import json

from app.repetition import (
    STOP_NOTICE,
    RepetitionGuard,
    clean_history,
    dedupe_sentences,
    guard_sse,
)

LOOP = "**광화문**은 **서울의 역사적인 문**으로, **한국의 중심부**에 위치한 **역사적 기념물**입니다. "
NORMAL = (
    "등록된 명소 검색(RAG)에서 일치하는 명소를 찾지 못했습니다. 사진에는 전통 한옥 양식의 문루가 보이며, "
    "흰색 석축 앞으로 관광객들이 걸어가고 있습니다. 하늘은 맑고 햇빛이 강한 낮 시간대입니다. "
    "지붕은 곡선형이고 단청 장식이 선명하게 보입니다."
)


def feed_all(guard: RepetitionGuard, text: str, step: int = 7) -> bool:
    return any(guard.feed(text[i : i + step]) for i in range(0, len(text), step))


def test_guard_trips_on_a_loop():
    assert feed_all(RepetitionGuard(), "이 사진은 광화문입니다. " + LOOP * 5)


def test_guard_ignores_normal_text():
    assert not feed_all(RepetitionGuard(), NORMAL * 1)


def test_guard_stays_tripped():
    guard = RepetitionGuard()
    feed_all(guard, LOOP * 5)
    assert guard.feed("완전히 다른 내용")


def sse(*deltas: str) -> list[bytes]:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': d}}]}, ensure_ascii=False)}\n\n"
        for d in deltas
    ]
    return [line.encode() for line in [*lines, "data: [DONE]\n\n"]]


async def collect(chunks: list[bytes]) -> str:
    async def source():
        for chunk in chunks:
            yield chunk

    return b"".join([c async for c in guard_sse(source(), RepetitionGuard())]).decode()


def contents(stream: str) -> str:
    out = ""
    for line in stream.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            out += json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    return out


def test_guard_sse_passes_normal_stream_unchanged():
    chunks = sse(*NORMAL.split(" "))
    assert asyncio.run(collect(chunks)) == b"".join(chunks).decode()


def test_guard_sse_stops_a_looping_stream():
    out = asyncio.run(collect(sse(*[LOOP] * 20)))

    assert out.endswith("data: [DONE]\n\n")
    text = contents(out)
    assert text.endswith(STOP_NOTICE)
    assert text.count(LOOP) < 5


def test_guard_sse_handles_korean_split_across_chunks():
    raw = b"".join(sse("광화문입니다"))
    # Split inside a multi-byte character.
    chunks = [raw[:13], raw[13:]]
    assert "광화문입니다" in contents(asyncio.run(collect(chunks)))


def test_dedupe_sentences_keeps_first_occurrence():
    text = "이 사진은 광화문입니다.\n\n" + LOOP * 3 + "\n\n" + LOOP * 2

    assert dedupe_sentences(text) == "이 사진은 광화문입니다.\n\n" + LOOP.strip()


def test_clean_history_dedupes_assistant_answers_and_drops_stop_notice():
    looped = "광화문입니다. " + LOOP * 30 + STOP_NOTICE
    user = {"role": "user", "content": LOOP * 3}

    cleaned = clean_history([user, {"role": "assistant", "content": looped}], max_chars=1000)

    assert cleaned[0] == user  # user messages are left alone
    assert cleaned[1] == {"role": "assistant", "content": "광화문입니다. " + LOOP.strip()}


def test_clean_history_caps_long_assistant_answers():
    cleaned = clean_history([{"role": "assistant", "content": NORMAL}], max_chars=60)

    content = cleaned[0]["content"]
    assert content.endswith("…")
    assert content[:-1] == NORMAL[:60].rstrip()


def test_guard_trips_on_a_loop_shorter_than_the_window():
    # A 44-char sentence repeated: matches of the 50-char tail overlap each other.
    sentence = "광화문은 서울의 역사적인 문으로, 한국의 중심부에 위치한 역사적 기념물입니다. "
    guard = RepetitionGuard()
    copies = next(i for i in range(1, 20) if guard.feed(sentence))
    assert copies == 4

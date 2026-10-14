"""Guards against degenerate repetition loops in small VLMs.

Once a looped answer is in the conversation, the model copies it into every later answer,
so loops are cut while streaming and stripped from the history that is sent back.
"""

import codecs
import json
import re
from collections.abc import AsyncIterator

STOP_NOTICE = "\n\n(같은 내용이 반복되어 답변을 중단했습니다.)"


class RepetitionGuard:
    """Detects when the latest `window` characters already appeared `max_repeats` times."""

    def __init__(self, window: int = 50, max_repeats: int = 3):
        self.window = window
        self.max_repeats = max_repeats
        self.text = ""
        self.tripped = False

    def feed(self, delta: str) -> bool:
        """Adds generated text; returns True once the answer is looping."""
        self.text += delta
        if len(self.text) >= self.window * self.max_repeats:
            tail = self.text[-self.window :]
            self.tripped = self.tripped or _count_overlapping(self.text, tail) >= self.max_repeats
        return self.tripped


def _count_overlapping(text: str, sub: str) -> int:
    # str.count skips overlapping matches, which undercounts loops shorter than the window.
    count, start = 0, text.find(sub)
    while start != -1:
        count += 1
        start = text.find(sub, start + 1)
    return count


def _sse_line(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


async def guard_sse(chunks: AsyncIterator[bytes], guard: RepetitionGuard) -> AsyncIterator[bytes]:
    """Relays an OpenAI-style SSE stream line by line, ending it early if the answer loops."""
    # Incremental decoding: a multi-byte character (e.g. Korean) can be split across chunks.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        *lines, buffer = buffer.split("\n")
        for line in lines:
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                choices = json.loads(line[6:]).get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content") or ""
                if guard.feed(delta):
                    yield _sse_line(
                        {"choices": [{"index": 0, "delta": {"content": STOP_NOTICE}, "finish_reason": "stop"}]}
                    )
                    yield _sse_line("[DONE]")
                    return
            yield f"{line}\n".encode()
    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.encode()


def _sentence_key(sentence: str) -> str:
    return re.sub(r"[\s*_`#>-]+", "", sentence)


def dedupe_sentences(text: str) -> str:
    """Drops sentences that already appeared earlier in the text, keeping paragraph breaks."""
    seen: set[str] = set()
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n", paragraph):
            key = _sentence_key(sentence)
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(sentence.strip())
        if kept:
            paragraphs.append(" ".join(kept))
    return "\n\n".join(paragraphs)


def clean_history(messages: list[dict], max_chars: int) -> list[dict]:
    """Removes repetition and caps the length of earlier assistant answers.

    The model otherwise imitates a looped answer it sees in the history.
    """
    cleaned = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str):
            content = dedupe_sentences(content.replace(STOP_NOTICE, ""))
            if len(content) > max_chars:
                content = content[:max_chars].rstrip() + "…"
            message = {**message, "content": content}
        cleaned.append(message)
    return cleaned

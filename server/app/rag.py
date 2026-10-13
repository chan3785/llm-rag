import logging
from dataclasses import dataclass
from typing import Literal

from fastapi.concurrency import run_in_threadpool
from qdrant_client import AsyncQdrantClient

from app.classify import Classification, vote
from app.config import settings
from app.embedding import ClipEmbedder, decode_data_url

logger = logging.getLogger(__name__)

PREVIOUS_IMAGE_TEXT = "(이전에 올린 이미지)"

Reason = Literal["matched", "below_threshold", "no_image", "no_reference_data", "retrieval_failed"]


@dataclass(frozen=True)
class AnswerSource:
    """Which stage decided the answer.

    RAG decides whenever the landmark search ran: it either matched a landmark or concluded
    the photo is not a registered one. The VLM answers alone only when no search was possible.
    """

    reason: Reason
    classification: Classification | None = None

    @property
    def stage(self) -> Literal["rag", "vlm"]:
        return "rag" if self.reason in ("matched", "below_threshold") else "vlm"

    def to_dict(self) -> dict:
        data: dict = {"stage": self.stage, "reason": self.reason}
        c = self.classification
        if c is not None:
            data |= {
                "landmark": c.landmark,
                "candidate": c.candidate,
                "score": round(c.score, 4),
                "threshold": settings.classify_threshold,
                "votes": c.votes,
                "total": c.total,
            }
        return data


def find_last_image(messages: list[dict]) -> str | None:
    """Returns the data URL of the most recent image in an OpenAI-style message list."""
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if part.get("type") == "image_url":
                return part["image_url"]["url"]
    return None


def keep_latest_image(messages: list[dict]) -> list[dict]:
    """Replaces images in earlier turns with a placeholder.

    Small VLMs mix up multiple images in one conversation and start inventing places or
    repeating themselves, so only the most recent image is sent.
    """
    last_with_image = max(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
        ),
        default=None,
    )
    result = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if i != last_with_image and isinstance(content, list):
            content = [
                {"type": "text", "text": PREVIOUS_IMAGE_TEXT} if p.get("type") == "image_url" else p
                for p in content
            ]
            message = {**message, "content": content}
        result.append(message)
    return result


async def analyze(messages: list[dict], clip: ClipEmbedder, qdrant: AsyncQdrantClient) -> AnswerSource:
    """Classifies the latest image against the ingested landmarks."""
    image_url = find_last_image(messages)
    if image_url is None:
        return AnswerSource("no_image")
    try:
        image = decode_data_url(image_url)
        # Encoding is blocking CPU/GPU work; keep it off the event loop.
        [vector] = await run_in_threadpool(clip.embed_images, [image])
        result = await qdrant.query_points(
            settings.qdrant_collection,
            query=vector,
            limit=settings.classify_top_k,
            with_payload=True,
        )
    except Exception:
        logger.warning("Landmark classification failed; continuing without context", exc_info=True)
        return AnswerSource("retrieval_failed")

    classification = vote(result.points, settings.classify_threshold, settings.classify_per_landmark)
    logger.info("Classified image: %s", classification)
    if classification.candidate is None:
        return AnswerSource("no_reference_data")
    if classification.landmark is None:
        return AnswerSource("below_threshold", classification)
    return AnswerSource("matched", classification)


def context_message(source: AnswerSource) -> dict | None:
    c = source.classification
    if c is None:
        return None
    if source.reason == "below_threshold":
        # Give the model an explicit way to answer; "don't guess" alone makes small models
        # loop through "it's not X, it's not Y..." until the token limit.
        content = (
            "등록된 명소 검색(RAG) 결과, 사용자가 올린 사진과 일치하는 명소를 찾지 못했습니다 "
            f"(최고 유사도 {c.score:.2f}, 기준 {settings.classify_threshold:.2f}).\n"
            "답변 방법:\n"
            "1. 첫 문장에서 \"등록된 명소 검색(RAG)에서 일치하는 명소를 찾지 못했습니다.\"라고 말하세요.\n"
            "2. 장소 이름은 추측하지 마세요. 어디인지 모르면 모른다고 답하세요.\n"
            "3. 사진에 보이는 풍경(지형, 색, 시간대 등)을 2~3문장으로 설명하고 끝내세요."
        )
    else:
        content = (
            f"사용자가 올린 사진은 등록된 명소 '{c.landmark}'(으)로 분류되었습니다 "
            f"(유사도 {c.score:.2f}, 비슷한 사진 {c.total}장 중 {c.votes}장)."
        )
        if c.description:
            content += f"\n명소 정보: {c.description}"
        content += "\n명소 정보에 없는 내용을 물으면 추측하지 말고 모른다고 답하세요."
    return {"role": "system", "content": content}

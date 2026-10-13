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
    """Which stage decided the answer: RAG (a registered landmark matched) or the VLM alone."""

    reason: Reason
    classification: Classification | None = None

    @property
    def stage(self) -> Literal["rag", "vlm"]:
        return "rag" if self.reason == "matched" else "vlm"

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
        content = (
            "사용자가 올린 사진은 등록된 명소 중 어느 곳과도 충분히 비슷하지 않습니다 "
            f"(최고 유사도 {c.score:.2f}). 특정 명소라고 단정하지 마세요."
        )
    else:
        content = (
            f"사용자가 올린 사진은 등록된 명소 '{c.landmark}'(으)로 분류되었습니다 "
            f"(유사도 {c.score:.2f}, 비슷한 사진 {c.total}장 중 {c.votes}장)."
        )
        if c.description:
            content += f"\n명소 정보: {c.description}"
    return {"role": "system", "content": content}

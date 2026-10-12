import logging

from fastapi.concurrency import run_in_threadpool
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.embedding import ClipEmbedder, decode_data_url

logger = logging.getLogger(__name__)


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


async def build_context_message(
    image_url: str, clip: ClipEmbedder, qdrant: AsyncQdrantClient
) -> dict | None:
    """Finds stored images similar to `image_url` and returns them as a system message.

    Returns None when nothing relevant is found or retrieval fails, so chat keeps working
    without context (e.g. before any data has been ingested).
    """
    try:
        image = decode_data_url(image_url)
        # Encoding is blocking CPU/GPU work; keep it off the event loop.
        [vector] = await run_in_threadpool(clip.embed_images, [image])
        result = await qdrant.query_points(
            settings.qdrant_collection,
            query=vector,
            limit=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
            with_payload=True,
        )
    except Exception:
        logger.warning("Image retrieval failed; continuing without context", exc_info=True)
        return None

    lines = [
        f"- {p.payload.get('description', '')} (similarity {p.score:.2f})"
        for p in result.points
        if p.payload and p.payload.get("description")
    ]
    if not lines:
        return None
    logger.info("Retrieved %d similar images", len(lines))
    return {
        "role": "system",
        "content": "다음은 사용자가 올린 이미지와 비슷한 자료입니다. 관련이 있으면 답변에 참고하세요.\n"
        + "\n".join(lines),
    }

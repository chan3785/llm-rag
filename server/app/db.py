from fastapi import Request
from qdrant_client import AsyncQdrantClient


def create_qdrant(url: str, api_key: str | None) -> AsyncQdrantClient:
    # Compatibility is checked by /health instead of a blocking request at startup.
    return AsyncQdrantClient(url=url, api_key=api_key, check_compatibility=False)


async def check_qdrant(client: AsyncQdrantClient) -> str:
    """Returns "ok" or "error: <reason>"."""
    try:
        await client.get_collections()
        return "ok"
    except Exception as exc:
        return f"error: {exc.__class__.__name__}"


def get_qdrant(request: Request) -> AsyncQdrantClient:
    """FastAPI dependency: `qdrant: AsyncQdrantClient = Depends(get_qdrant)`."""
    return request.app.state.qdrant

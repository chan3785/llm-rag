from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient

from app import db
from app.config import settings
from app.llm import LlamaClient, get_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LlamaClient(settings.llama_server_url, settings.llm_timeout_seconds)
    app.state.qdrant = db.create_qdrant(settings.qdrant_url, settings.qdrant_api_key)
    yield
    await app.state.llm.close()
    await app.state.qdrant.close()


app = FastAPI(title="llm-rag server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(
    llm: LlamaClient = Depends(get_llm),
    qdrant: AsyncQdrantClient = Depends(db.get_qdrant),
):
    return {"status": "ok", "llm": await llm.check(), "db": await db.check_qdrant(qdrant)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, llm: LlamaClient = Depends(get_llm)):
    """Relays an OpenAI-style chat request to llama-server, streaming the response as-is."""
    body = await request.json()
    try:
        upstream = await llm.open_chat_stream(body)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach llama-server ({settings.llama_server_url}): {exc.__class__.__name__}",
        )

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )

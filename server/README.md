# llm-rag server

FastAPI server that sits between the Next.js web app, llama-server and the Qdrant vector DB.

```
[Ingestion]  image folder → CLIP image embedding → Qdrant (vector + description)

[Chat]       Next.js (/api/chat) → FastAPI (:8000) ─┬─ last uploaded image → CLIP → Qdrant search
                                                    └─ matches added as a system message → llama-server (:8080)
```

## Run

```bash
cd server
cp .env.example .env   # edit as needed
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status of llama-server (`llm`) and Qdrant (`db`): `ok` / `error: ...` |
| POST | `/v1/chat/completions` | OpenAI-compatible chat request, relayed to llama-server (streaming included) |

API docs: http://localhost:8000/docs

## Qdrant

The client is created in `app/db.py` and opened/closed with the app.

1. Start Qdrant from the repo root: `docker compose up -d` (dashboard: http://localhost:6333/dashboard).
2. Set `QDRANT_URL` (and `QDRANT_API_KEY` if enabled) in `.env`. Default: `http://localhost:6333`.
3. Check `GET /health` shows `"db": "ok"`.
4. Use the client in routes:

```python
from fastapi import Depends
from qdrant_client import AsyncQdrantClient
from app.db import get_qdrant

@app.get("/collections")
async def list_collections(qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    return await qdrant.get_collections()
```

## Ingest images (CLIP)

Put images in a folder. A sidecar `.txt` with the same name becomes the description
the LLM sees (e.g. `cat.jpg` + `cat.txt`).

```bash
uv run python -m scripts.ingest ./data/images            # re-running updates the same files in place
uv run python -m scripts.ingest ./data/images --recreate # drop and rebuild the collection
```

The first run downloads the CLIP model (~600 MB, cached in `~/.cache/huggingface`).
Ingestion and chat must use the same `CLIP_MODEL`.

## Retrieval at chat time

When a chat request contains an image, `app/rag.py` embeds the most recent image,
queries Qdrant (`RAG_TOP_K`, `RAG_SCORE_THRESHOLD`) and prepends the descriptions of matches
as a system message. If nothing passes the threshold, or Qdrant/the collection is unavailable,
the request is forwarded unchanged.

`RAG_SCORE_THRESHOLD` is cosine similarity and depends on your data: CLIP scores unrelated
photos of a similar style around 0.7–0.8. Check scores against real data and adjust.

## Test

```bash
uv run pytest   # CLIP is replaced with a fake, so no model download is needed
```

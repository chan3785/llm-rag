# llm-rag server

FastAPI server that sits between the Next.js web app, llama-server and the Qdrant vector DB.

```
Next.js (/api/chat) → FastAPI (:8000) → llama-server (:8080)
                                      ↘ Qdrant (:6333)
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

## Test

```bash
uv run pytest
```

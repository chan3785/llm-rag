# llm-rag server

FastAPI server that sits between the Next.js web app, llama-server and the database.

```
Next.js (/api/chat) → FastAPI (:8000) → llama-server (:8080)
                                      ↘ DB (DATABASE_URL)
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
| GET | `/health` | Status of llama-server (`llm`) and the database (`db`: `ok` / `not_configured` / `error: ...`) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat request, relayed to llama-server (streaming included) |

API docs: http://localhost:8000/docs

## Database

The connection is set up in `app/db.py` and is skipped while `DATABASE_URL` is unset.

1. Start the database (e.g. Postgres with Docker).
2. Set `DATABASE_URL` in `.env`, e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/llm_rag`.
   For another database, install its async driver and change the URL scheme.
3. Check `GET /health` shows `"db": "ok"`.
4. Define models by subclassing `app.db.Base`, and use a session in routes:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session

@app.get("/items")
async def list_items(session: AsyncSession = Depends(get_session)):
    ...
```

## Test

```bash
uv run pytest
```

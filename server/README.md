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

## Ingest landmark images (CLIP)

One folder per landmark; the folder name is the label. An optional `_info.txt` in the folder
is the landmark description given to the LLM.

```
data/landmarks/
├── 경복궁/
│   ├── _info.txt
│   ├── 001.jpg
│   └── ...
└── 불국사/...
```

```bash
uv run python -m scripts.ingest ./data/landmarks            # re-running updates the same files in place
uv run python -m scripts.ingest ./data/landmarks --recreate # drop and rebuild the collection
```

The first run downloads the CLIP model (~600 MB, cached in `~/.cache/huggingface`).
Ingestion and chat must use the same `CLIP_MODEL`; after changing it, re-ingest with `--recreate`.

## Classification at chat time

When a chat request contains an image, `app/rag.py` embeds the most recent image and retrieves
the `CLASSIFY_TOP_K` nearest reference images. Landmarks are compared by the mean similarity of
their `CLASSIFY_PER_LANDMARK` closest images (`app/classify.py`), so landmarks with many photos
don't win by volume. If the winner's best similarity is below `CLASSIFY_THRESHOLD`, the photo is
treated as an unregistered place.

The result is prepended to the conversation as a system message, and returned in the
`X-Answer-Source` response header, which the web app shows above each answer:

| `stage` | `reason` | Meaning |
|---|---|---|
| `rag` | `matched` | A registered landmark matched; the answer is grounded in its description |
| `vlm` | `below_threshold` | Closest landmark (`candidate`) scored below the threshold; the VLM answers on its own |
| `vlm` | `no_image` / `no_reference_data` / `retrieval_failed` | No classification was possible |

Only the latest image in the conversation is sent to the VLM (earlier ones are replaced by a
placeholder), since small VLMs mix up multiple images and start repeating themselves.

## Evaluate

Keep held-out photos (not ingested) in the same layout, plus `_unknown/` for unregistered places:

```bash
uv run python -m scripts.evaluate ./data/eval
```

It prints per-image predictions and accuracy across thresholds to help pick `CLASSIFY_THRESHOLD`.

## Test

```bash
uv run pytest   # CLIP is replaced with a fake, so no model download is needed
```

# llm-rag

Landmark chat: upload a photo, CLIP classifies it against registered landmarks in Qdrant,
and a VLM (llama-server) answers using the result.

```
Web (Next.js :3000) → API (FastAPI :8000) ─┬─ CLIP → Qdrant (:6333)
                                           └─ llama-server (:8080)
```

## Run

```bash
make up      # Qdrant (Docker) + llama-server + API server, logs in one terminal; Ctrl+C stops all
yarn dev     # web app, in another terminal → http://localhost:3000
```

- `make status` shows the health of each service; `make down` stops the Qdrant container.
- llama-server and the API run natively (not in Docker) so they can use the Mac GPU.
- `make up` refuses to start if port 8080 or 8000 is already taken, and prints the process using it.
- Override settings per run, e.g. `make up LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M LLM_CTX=16384`.

Requirements: Docker, [`llama`](https://github.com/ggml-org/llama.cpp) on `PATH`, [`uv`](https://docs.astral.sh/uv/), Yarn.

## More

- API server, ingestion and evaluation: [`server/README.md`](server/README.md)

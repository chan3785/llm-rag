# Local dev stack: Qdrant (Docker) + llama-server and the FastAPI server (native, so both use the GPU).
# Override any variable, e.g. `make up LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M`.

LLAMA     ?= llama
LLM_MODEL ?= Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M
LLM_CTX   ?= 8192
LLM_PORT  ?= 8080
API_PORT  ?= 8000
export LLAMA LLM_MODEL LLM_CTX LLM_PORT API_PORT

.PHONY: help up down status

help:
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

up: ## Start Qdrant, llama-server and the API server together (Ctrl+C stops all)
	@for port in $(LLM_PORT) $(API_PORT); do \
		if lsof -nP -iTCP:$$port -sTCP:LISTEN >/dev/null; then \
			echo "Port $$port is already in use:"; lsof -nP -iTCP:$$port -sTCP:LISTEN; exit 1; \
		fi; \
	done
	uvx honcho -f Procfile.dev start

down: ## Stop the Qdrant container (llama-server and the API stop with Ctrl+C)
	docker compose stop qdrant

status: ## Show health of each service
	@printf "qdrant (6333):        "; curl -sf -m 2 localhost:6333/healthz >/dev/null && echo ok || echo down
	@printf "llm    ($(LLM_PORT)):        "; curl -sf -m 2 localhost:$(LLM_PORT)/health >/dev/null && echo ok || echo down
	@printf "api    ($(API_PORT)):        "; curl -sf -m 2 localhost:$(API_PORT)/health || echo down; echo

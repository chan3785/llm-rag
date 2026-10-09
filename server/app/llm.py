import httpx
from fastapi import Request


class LlamaClient:
    """Thin async client for llama-server's OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout: float):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def check(self) -> str:
        try:
            res = await self._client.get("/health", timeout=3.0)
            return "ok" if res.is_success else f"error: HTTP {res.status_code}"
        except httpx.HTTPError as exc:
            return f"error: {exc.__class__.__name__}"

    async def open_chat_stream(self, body: dict) -> httpx.Response:
        """Sends a chat completion request and returns the unread response.

        The caller must close it (`await response.aclose()`).
        """
        request = self._client.build_request("POST", "/v1/chat/completions", json=body)
        return await self._client.send(request, stream=True)


def get_llm(request: Request) -> LlamaClient:
    return request.app.state.llm

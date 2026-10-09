const API_SERVER_URL = process.env.API_SERVER_URL ?? "http://localhost:8000";

// Forwards an OpenAI-style chat request to the FastAPI server and streams the SSE response back.
export async function POST(request: Request) {
  const { messages } = await request.json();

  let upstream: Response;
  try {
    upstream = await fetch(`${API_SERVER_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, stream: true }),
      signal: request.signal,
    });
  } catch {
    return Response.json(
      { error: `API 서버(${API_SERVER_URL})에 연결할 수 없습니다.` },
      { status: 502 },
    );
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text();
    return Response.json(
      { error: `API 서버 오류 (${upstream.status}): ${detail}` },
      { status: 502 },
    );
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    },
  });
}

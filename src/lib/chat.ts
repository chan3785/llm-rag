export type Message = {
  id: number;
  role: "user" | "assistant";
  text: string;
  imageUrl?: string;
};

type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type ApiMessage = { role: Message["role"]; content: string | ContentPart[] };

export function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function toApiMessage(m: Message): ApiMessage {
  if (!m.imageUrl) return { role: m.role, content: m.text };
  const parts: ContentPart[] = [{ type: "image_url", image_url: { url: m.imageUrl } }];
  if (m.text) parts.push({ type: "text", text: m.text });
  return { role: m.role, content: parts };
}

// Sends the conversation to /api/chat and calls onDelta with each streamed token.
export async function streamChat(
  messages: Message[],
  onDelta: (text: string) => void,
  signal?: AbortSignal,
) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: messages.map(toApiMessage) }),
    signal,
  });

  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.error ?? `요청 실패 (${res.status})`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") return;
      const delta = JSON.parse(data).choices?.[0]?.delta?.content;
      if (delta) onDelta(delta);
    }
  }
}

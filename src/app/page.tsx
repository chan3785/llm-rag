"use client";

import { useEffect, useRef, useState } from "react";

type Message = {
  id: number;
  text: string;
  imageUrl?: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<{ file: File; url: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleImageChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (image) URL.revokeObjectURL(image.url);
    setImage({ file, url: URL.createObjectURL(file) });
  }

  function removeImage() {
    if (image) URL.revokeObjectURL(image.url);
    setImage(null);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text && !image) return;
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), text, imageUrl: image?.url },
    ]);
    setInput("");
    // The object URL now belongs to the message, so don't revoke it here.
    setImage(null);
  }

  return (
    <main className="mx-auto flex h-dvh w-full max-w-2xl flex-col">
      <header className="border-b border-black/10 px-4 py-3 dark:border-white/15">
        <h1 className="text-lg font-semibold">Chat</h1>
      </header>

      <section className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <p className="mt-10 text-center text-sm text-gray-500">
            메시지를 입력하거나 이미지를 업로드하세요.
          </p>
        )}
        {messages.map((m) => (
          <div key={m.id} className="flex justify-end">
            <div className="max-w-[80%] space-y-2 rounded-2xl bg-blue-600 px-4 py-2 text-white">
              {m.imageUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={m.imageUrl}
                  alt="업로드한 이미지"
                  className="max-h-64 rounded-lg"
                />
              )}
              {m.text && <p className="whitespace-pre-wrap break-words">{m.text}</p>}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </section>

      <form
        onSubmit={handleSubmit}
        className="border-t border-black/10 px-4 py-3 dark:border-white/15"
      >
        {image && (
          <div className="relative mb-2 inline-block">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={image.url}
              alt={image.file.name}
              className="h-20 rounded-lg border border-black/10 object-cover"
            />
            <button
              type="button"
              onClick={removeImage}
              aria-label="이미지 제거"
              className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-gray-800 text-xs text-white"
            >
              ✕
            </button>
          </div>
        )}
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0 rounded-full border border-black/15 px-3 py-2 text-sm hover:bg-black/5 dark:border-white/20 dark:hover:bg-white/10"
          >
            이미지 업로드
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="메시지를 입력하세요"
            className="min-w-0 flex-1 rounded-full border border-black/15 bg-transparent px-4 py-2 outline-none focus:border-blue-500 dark:border-white/20"
          />
          <button
            type="submit"
            disabled={!input.trim() && !image}
            className="shrink-0 rounded-full bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            전송
          </button>
        </div>
      </form>
    </main>
  );
}

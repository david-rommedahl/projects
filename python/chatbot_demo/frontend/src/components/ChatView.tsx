import { useEffect, useRef, useState } from "react";
import type { Message } from "../types";

interface Props {
  messages: Message[];
  streaming: boolean;
  error: string | null;
  onSend: (question: string) => void;
}

export function ChatView({ messages, streaming, error, onSend }: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // Keep the latest message in view as tokens stream in.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function submit() {
    const question = input.trim();
    if (!question || streaming) return;
    onSend(question);
    setInput("");
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <main className="chat">
      <div className="messages">
        {messages.length === 0 && <p className="empty">Ask something to start the conversation.</p>}
        {messages.map((m, i) => (
          <div key={i} className={"message " + m.role}>
            <div className="role">{m.role}</div>
            <div className="content">
              {m.content}
              {streaming && i === messages.length - 1 && m.role === "assistant" && <span className="cursor">▋</span>}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error banner">{error}</div>}

      <div className="composer">
        <textarea
          rows={2}
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={streaming}
        />
        <button onClick={submit} disabled={streaming || !input.trim()}>
          {streaming ? "…" : "Send"}
        </button>
      </div>
    </main>
  );
}

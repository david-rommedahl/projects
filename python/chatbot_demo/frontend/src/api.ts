import { getApiKey } from "./auth";
import type { ConversationSummary, Message, StreamEvent } from "./types";

const API = "/api/v1";

/** Thrown for non-2xx responses so callers can branch on status (e.g. 401). */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

/** Register a user by email; returns the freshly issued API key (shown once). */
export async function register(email: string): Promise<string> {
  const res = await fetch(`${API}/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) throw new ApiError(res.status, `Registration failed (${res.status})`);
  const data = await res.json();
  return data.api_key as string;
}

/** List the authenticated user's conversations, newest first. */
export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await fetch(`${API}/conversations`, { headers: authHeaders() });
  if (!res.ok) throw new ApiError(res.status, `Could not load conversations (${res.status})`);
  const data = await res.json();
  return data.conversations as ConversationSummary[];
}

/** Fetch the transcript for one conversation. */
export async function getMessages(sessionId: string): Promise<Message[]> {
  const res = await fetch(`${API}/conversations/${encodeURIComponent(sessionId)}/messages`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new ApiError(res.status, `Could not load messages (${res.status})`);
  const data = await res.json();
  return data.messages as Message[];
}

export interface ChatHandlers {
  /** A chunk of assistant text arrived. */
  onToken: (content: string) => void;
  /** The server reported a mid-stream generation error. */
  onError: (message: string) => void;
  /** The resolved session token (from the X-Session-Id header). */
  onSessionId: (sessionId: string) => void;
}

/**
 * Send a question and consume the NDJSON event stream from POST /chat.
 *
 * The response is newline-delimited JSON, not SSE, so we read the body as a
 * stream and parse one event object per line. Resolves when the stream ends
 * (a `done` event); rejects only on transport/HTTP errors before streaming.
 */
export async function streamChat(
  question: string,
  sessionId: string | null,
  handlers: ChatHandlers,
): Promise<void> {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!res.ok) throw new ApiError(res.status, `Chat request failed (${res.status})`);

  const sid = res.headers.get("X-Session-Id");
  if (sid) handlers.onSessionId(sid);

  if (!res.body) throw new Error("Response has no body to stream");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex: number;
    while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;
      const event = JSON.parse(line) as StreamEvent;
      if (event.type === "token") handlers.onToken(event.content);
      else if (event.type === "error") handlers.onError(event.content);
      // `done` needs no handling — the loop simply ends.
    }
  }
}

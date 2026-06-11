export interface ConversationSummary {
  session_id: string;
  created_at: string;
}

export interface Message {
  role: string;
  content: string;
}

// Mirrors the backend's NDJSON stream events from POST /chat.
export type StreamEvent =
  | { type: "token"; content: string }
  | { type: "error"; content: string }
  | { type: "done" };

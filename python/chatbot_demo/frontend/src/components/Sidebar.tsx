import { useState } from "react";
import type { ConversationSummary } from "../types";

interface Props {
  apiKey: string;
  conversations: ConversationSummary[];
  currentSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNewChat: () => void;
  onLogout: () => void;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function Sidebar({ apiKey, conversations, currentSessionId, onSelect, onNewChat, onLogout }: Props) {
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);

  async function copyKey() {
    try {
      await navigator.clipboard.writeText(apiKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. non-secure context) — reveal lets the
      // user copy manually instead.
      setRevealed(true);
    }
  }

  return (
    <aside className="sidebar">
      <button className="new-chat" onClick={onNewChat}>
        + New chat
      </button>

      <nav className="conversation-list">
        {conversations.length === 0 && <p className="empty">No conversations yet.</p>}
        {conversations.map((c) => (
          <button
            key={c.session_id}
            className={"conversation" + (c.session_id === currentSessionId ? " active" : "")}
            onClick={() => onSelect(c.session_id)}
            title={c.session_id}
          >
            <span className="conversation-title">{c.title || "Untitled conversation"}</span>
            <span className="conversation-date">{formatDate(c.created_at)}</span>
          </button>
        ))}
      </nav>

      <div className="account">
        <div className="account-row">
          <span className="account-label">API key</span>
          <div className="account-actions">
            <button onClick={() => setRevealed((r) => !r)}>{revealed ? "Hide" : "Show"}</button>
            <button onClick={copyKey}>{copied ? "Copied" : "Copy"}</button>
          </div>
        </div>
        {revealed && <code className="key-box">{apiKey}</code>}
        <p className="hint">Stored on this device only — keep it safe; it can't be recovered elsewhere.</p>
      </div>

      <button className="logout" onClick={onLogout}>
        Sign out
      </button>
    </aside>
  );
}

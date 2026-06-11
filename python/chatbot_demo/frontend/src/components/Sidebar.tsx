import type { ConversationSummary } from "../types";

interface Props {
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

export function Sidebar({ conversations, currentSessionId, onSelect, onNewChat, onLogout }: Props) {
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
            <span className="conversation-id">{c.session_id.slice(0, 8)}…</span>
            <span className="conversation-date">{formatDate(c.created_at)}</span>
          </button>
        ))}
      </nav>

      <button className="logout" onClick={onLogout}>
        Sign out
      </button>
    </aside>
  );
}

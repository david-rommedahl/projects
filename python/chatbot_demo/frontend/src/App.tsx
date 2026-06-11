import { useCallback, useEffect, useState } from "react";
import { ApiError, getMessages, listConversations, streamChat } from "./api";
import { clearApiKey, getApiKey, setApiKey } from "./auth";
import { ChatView } from "./components/ChatView";
import { KeyGate } from "./components/KeyGate";
import { Sidebar } from "./components/Sidebar";
import type { ConversationSummary, Message } from "./types";

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean>(() => getApiKey() !== null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const logout = useCallback(() => {
    clearApiKey();
    setAuthenticated(false);
    setConversations([]);
    setMessages([]);
    setSessionId(null);
    setError(null);
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await listConversations());
    } catch (err) {
      // An expired/invalid stored key surfaces here as a 401 — sign out.
      if (err instanceof ApiError && err.status === 401) logout();
      else setError(err instanceof Error ? err.message : String(err));
    }
  }, [logout]);

  useEffect(() => {
    if (authenticated) void refreshConversations();
  }, [authenticated, refreshConversations]);

  function handleAuthenticated(key: string) {
    setApiKey(key);
    setAuthenticated(true);
  }

  function startNewChat() {
    setSessionId(null);
    setMessages([]);
    setError(null);
  }

  async function selectConversation(id: string) {
    setError(null);
    setSessionId(id);
    try {
      setMessages(await getMessages(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function send(question: string) {
    setError(null);
    setStreaming(true);
    const wasNewConversation = sessionId === null;

    // Optimistically render the user's message and an empty assistant message
    // that we append streamed tokens to.
    setMessages((prev) => [...prev, { role: "user", content: question }, { role: "assistant", content: "" }]);

    const appendToAssistant = (text: string) =>
      setMessages((prev) => {
        const next = prev.slice();
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, content: last.content + text };
        return next;
      });

    try {
      await streamChat(question, sessionId, {
        onSessionId: setSessionId,
        onToken: appendToAssistant,
        onError: setError,
      });
      // A brand-new conversation now exists server-side — show it in the sidebar.
      if (wasNewConversation) void refreshConversations();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) logout();
      else setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStreaming(false);
    }
  }

  if (!authenticated) return <KeyGate onAuthenticated={handleAuthenticated} />;

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentSessionId={sessionId}
        onSelect={selectConversation}
        onNewChat={startNewChat}
        onLogout={logout}
      />
      <ChatView messages={messages} streaming={streaming} error={error} onSend={send} />
    </div>
  );
}

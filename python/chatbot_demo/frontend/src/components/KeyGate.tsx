import { useState } from "react";
import { register } from "../api";

interface Props {
  /** Called with a valid API key once the user registers or pastes one. */
  onAuthenticated: (apiKey: string) => void;
}

/**
 * Sign-in gate. Two ways in:
 *  - Register a new email -> the backend returns an API key we keep.
 *  - Paste an existing key.
 */
export function KeyGate({ onAuthenticated }: Props) {
  const [mode, setMode] = useState<"register" | "paste">("register");
  const [email, setEmail] = useState("");
  const [pastedKey, setPastedKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const key = await register(email.trim());
      onAuthenticated(key);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function handlePaste(e: React.FormEvent) {
    e.preventDefault();
    const key = pastedKey.trim();
    if (key) onAuthenticated(key);
  }

  return (
    <div className="gate">
      <div className="gate-card">
        <h1>Chat demo</h1>
        <div className="gate-tabs">
          <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>
            Register
          </button>
          <button className={mode === "paste" ? "active" : ""} onClick={() => setMode("paste")}>
            I have a key
          </button>
        </div>

        {mode === "register" ? (
          <form onSubmit={handleRegister}>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              required
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button type="submit" disabled={busy || !email.trim()}>
              {busy ? "Registering…" : "Register & get a key"}
            </button>
            <p className="hint">An API key is generated and stored locally. Re-registering issues a new key.</p>
          </form>
        ) : (
          <form onSubmit={handlePaste}>
            <label htmlFor="key">API key</label>
            <input
              id="key"
              type="password"
              required
              placeholder="paste your API key"
              value={pastedKey}
              onChange={(e) => setPastedKey(e.target.value)}
            />
            <button type="submit" disabled={!pastedKey.trim()}>
              Continue
            </button>
          </form>
        )}

        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}

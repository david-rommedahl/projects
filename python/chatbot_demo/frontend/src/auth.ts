// The API key is the only credential. We persist it in localStorage so a reload
// keeps the user signed in. (A demo-grade choice — fine here, but localStorage is
// readable by any script on the origin.)
const STORAGE_KEY = "chat_api_key";

export function getApiKey(): string | null {
  return localStorage.getItem(STORAGE_KEY);
}

export function setApiKey(key: string): void {
  localStorage.setItem(STORAGE_KEY, key);
}

export function clearApiKey(): void {
  localStorage.removeItem(STORAGE_KEY);
}

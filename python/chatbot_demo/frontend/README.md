# chat-frontend

A minimal React + Vite + TypeScript UI for the chat service: register for an API
key, chat with streaming responses, and browse past conversations.

## Run

The backend must be running on `localhost:8000` (see `../backend`). Then:

```bash
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173).

API calls go to relative `/api/*` paths, which Vite proxies to the backend
(see [vite.config.ts](vite.config.ts)) — so the browser makes same-origin
requests and no CORS config is needed on the backend.

## How it works

- **Auth** ([src/auth.ts](src/auth.ts)) — the API key is the only credential.
  Register with an email (`POST /users`) to get one, or paste an existing key.
  It's kept in `localStorage` and sent as `Authorization: Bearer <key>`.
- **Streaming** ([src/api.ts](src/api.ts)) — `POST /chat` returns newline-delimited
  JSON, not SSE, so the response body is read as a stream and parsed one event
  (`token` / `error` / `done`) per line.
- **Sessions** — the server returns `X-Session-Id`; the UI echoes it back to
  continue a conversation. `GET /conversations` populates the sidebar and
  `GET /conversations/{id}/messages` loads history.

## Scripts

- `npm run dev` — dev server with hot reload + API proxy
- `npm run build` — typecheck (`tsc`) and production build
- `npm run typecheck` — types only
- `npm run preview` — serve the production build

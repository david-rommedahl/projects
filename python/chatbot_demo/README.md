# chatbot_demo

A small but complete AI chat service: a **FastAPI** backend (LangChain + an
OpenAI‑compatible model, conversations persisted in **PostgreSQL**) and a
**React** chat UI. The whole thing runs with one `docker compose` command.

- Register with an email → get an API key.
- Chat with streaming responses; conversations are saved per user.
- Browse and revisit past conversations.

---

## Quick start (Docker — recommended)

You need [Docker](https://docs.docker.com/get-docker/) (with Compose) and an
OpenAI API key.

```bash
# 1. Create your env file from the template
cp .env.example .env

# 2. Edit .env and set, at minimum:
#      POSTGRES_PASSWORD   (any value)
#      OPENAI_API_KEY      (your key)

# 3. Build and start everything (backend, database, frontend)
docker compose up --build -d
```

Then open **http://localhost:8080** and register with any email — you'll get an
API key (stored in your browser) and land in the chat.

That's it. The backend runs its database migrations automatically on startup, so
a fresh database is ready to use.

To watch logs or stop:

```bash
docker compose logs -f          # follow logs
docker compose down             # stop (keeps your data)
docker compose down -v          # stop and wipe the database
```

---

## Configuration

All configuration is via `.env` (copied from [`.env.example`](.env.example)):

| Variable           | Required | Description                                                             |
| ------------------ | -------- | ----------------------------------------------------------------------- |
| `POSTGRES_PASSWORD`| yes      | Password for the bundled PostgreSQL.                                    |
| `POSTGRES_USER`    | –        | Defaults to `postgres`.                                                 |
| `POSTGRES_DB`      | –        | Defaults to `chat_service`.                                             |
| `OPENAI_API_KEY`   | yes      | API key for the LLM provider.                                           |
| `CHAT_MODEL`       | –        | Model/deployment name (e.g. `gpt-4o-mini`).                             |
| `OPENAI_BASE_URL`  | –        | Leave unset for OpenAI. Set it to point at an OpenAI‑compatible endpoint (e.g. an Azure AI Foundry deployment) instead. |
| `LOG_LEVEL`        | –        | `DEBUG` / `INFO` / … Defaults to `INFO`.                                |

The same client works against OpenAI directly **or** an internal Azure
deployment — only `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `CHAT_MODEL` differ.
See the comments in [`.env.example`](.env.example) for the Azure variant.

---

## How it fits together

```
                http://localhost:8080
                        │
                ┌───────▼────────┐      /api/* proxied
                │   frontend     │────────────────────┐
                │ (nginx + SPA)  │                     │
                └────────────────┘                     ▼
                                              ┌──────────────────┐
                                              │     backend      │
                                              │   (FastAPI)      │
                                              └─────────┬────────┘
                                                        │
                                                ┌───────▼────────┐
                                                │   PostgreSQL   │
                                                └────────────────┘
```

- **frontend** ([`frontend/`](frontend/)) — React + Vite app, built to static
  files and served by nginx, which also reverse‑proxies `/api` to the backend
  (so the browser sees one origin, no CORS).
- **backend** ([`backend/`](backend/)) — FastAPI service. Authenticates via an
  API key, streams chat responses, and persists conversations in PostgreSQL via
  a LangGraph checkpointer.
- **db** — PostgreSQL 17.

| Service  | URL                     |
| -------- | ----------------------- |
| Frontend | http://localhost:8080   |
| Backend  | http://localhost:8000   |
| Postgres | localhost:5432          |

### API at a glance

All endpoints are under `/api/v1`. Authenticate with `Authorization: Bearer <api_key>`.

| Method & path                              | Purpose                                        |
| ------------------------------------------ | ---------------------------------------------- |
| `POST /users`                              | Register an email → returns an API key.        |
| `POST /chat`                               | Send a question; streams the answer (NDJSON).  |
| `GET  /conversations`                      | List your conversations.                       |
| `GET  /conversations/{id}/messages`        | Fetch a conversation's transcript.             |
| `GET  /ping`                               | Health check.                                  |

Interactive API docs are at http://localhost:8000/docs.

---

## Authentication

The only credential is an **API key**, sent as `Authorization: Bearer <key>`.

- **Registering** (`POST /users` with an email) generates a key and returns it
  **once**. The frontend stores it in `localStorage`; you can also reveal/copy it
  later from the sidebar.
- Only a **SHA‑256 hash** of the key is stored — never the raw key. Authentication
  hashes the presented key and looks up the matching user. A fast hash is
  appropriate here because keys are high‑entropy random tokens (unlike passwords,
  which would need a slow KDF).
- Because the raw key isn't stored, it can't be recovered. Registering an email
  that already exists **rotates** the key (issues a new one) rather than erroring.
- All authentication lives behind one dependency (`get_current_user`). It resolves
  a key to a stable, non‑secret **`user_id`**, which is what everything else is
  scoped by. Swapping in real auth later (JWT, OAuth, …) only touches that seam —
  no endpoint changes.

> This is demo‑grade: registration is open and unauthenticated, there's no rate
> limiting, and `localStorage` is readable by any script on the origin. Fine for a
> demo; hardening these is the obvious next step for production.
> For a production version, SSO login through EntraID or similar is a clean choice.

---

## Architecture & design decisions

The guiding idea is **clear seams**: each external concern (the LLM provider,
auth, transcript storage) sits behind one swappable boundary, so the demo stays
simple but isn't painted into a corner.

- **One LLM client, provider‑agnostic.** A single `get_chat_model()` factory
  returns a `ChatOpenAI` configured from env. Because Azure AI Foundry exposes an
  OpenAI‑compatible surface, the *same* client serves both OpenAI and an internal
  Azure deployment — only `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `CHAT_MODEL`
  change, with no provider branching. The factory is the seam for a future
  provider registry. Requests carry a timeout and a couple of retries so a stalled
  upstream fails fast.

- **Identity vs. ownership are separate.** The API key is the *credential*; a
  stable `user_id` is the *identity*. Data is partitioned by `user_id`, never by
  the secret — so rotating a key doesn't orphan a user's conversations.

- **Two layers of conversation storage.** A `conversation` table (owned by us)
  maps each session token to its `owner_id` — this is the ownership/index layer
  used for authorization and listing. The actual message transcript lives in a
  **LangGraph Postgres checkpointer**, keyed by `thread_id == session_id`. Keeping
  them separate means listing, ownership checks, and titles are cheap SQL, while
  the transcript store stays an opaque, swappable component.

- **Privacy by construction.** Every conversation endpoint authorizes ownership
  first and returns **404** (not 403) for both "doesn't exist" and "belongs to
  someone else", so the API never leaks which session tokens exist. A user can
  only ever see their own conversations.

- **Sessions via a round‑tripped token.** The server mints a `session_id` on the
  first message and returns it in the `X-Session-Id` header; the client echoes it
  back to continue the conversation. History is carried forward by the
  checkpointer, so follow‑up turns retain context with no client‑side state.

- **Streaming as NDJSON, not SSE.** `POST /chat` streams newline‑delimited JSON
  events (`token` / `error` / `done`). This rides on a normal POST (SSE is
  GET‑only) and is structured, so a mid‑stream model failure is reported in‑band
  as an `error` event — the HTTP status is already `200` once streaming starts —
  and a `done` event always terminates the stream, letting the client distinguish
  completion from a dropped connection. (Buffering is deliberately avoided
  end‑to‑end: no buffering middleware on the backend, `proxy_buffering off` in
  nginx, and the browser parses the byte stream incrementally.)

- **Titles without an LLM.** A conversation's title is derived from the first
  message (whitespace‑collapsed, truncated) at creation time — no summarization
  call, no extra latency or cost.

- **Config split so migrations stay lightweight.** `DatabaseConfig` (DB‑only) is
  separate from the full runtime `Config`, and the runtime config is resolved
  lazily. This lets Alembic compose a connection URL without requiring the
  `OPENAI_API_KEY`, and keeps secrets out of the migration path.

- **Same‑origin frontend, no CORS.** In dev, Vite proxies `/api` to the backend;
  in Docker, nginx serves the built SPA and reverse‑proxies `/api`. Either way the
  browser talks to one origin, so no CORS configuration is needed.

- **Migrations run on startup.** The backend container's entrypoint runs
  `alembic upgrade head` (idempotent) before serving, so a fresh database is
  schema‑ready from a single `docker compose up`. The checkpointer's own tables
  are created at runtime by the checkpointer and are intentionally excluded from
  Alembic's autogenerate.

---

## Local development (without Docker)

Useful for hot‑reload and running tests. Requires
[uv](https://docs.astral.sh/uv/) and [Node.js](https://nodejs.org/) 20+.

You'll need a PostgreSQL to point at — the easiest is just the bundled one:

```bash
docker compose up -d db
```

**Backend** (from [`backend/`](backend/)):

```bash
cp .env.example .env          # set OPENAI_API_KEY, POSTGRES_PASSWORD; POSTGRES_HOST=localhost
uv sync                       # install dependencies
uv run alembic upgrade head   # create the database tables
uv run uvicorn chat_service.asgi:app --reload
```

**Frontend** (from [`frontend/`](frontend/)):

```bash
npm install
npm run dev                   # http://localhost:5173, proxies /api to localhost:8000
```

**Tests / checks** (from [`backend/`](backend/)):

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

See [`backend/README.md`](backend/README.md) and
[`frontend/README.md`](frontend/README.md) for more detail.

---

## Project layout

```
chatbot_demo/
├── docker-compose.yml     # db + backend + frontend
├── .env.example           # configuration template
├── backend/               # FastAPI service (uv, SQLAlchemy, Alembic, LangChain)
└── frontend/              # React + Vite chat UI (served by nginx in Docker)
```

# Plan: scaffold `chatbot_demo/backend`

Scaffolding plan for a simple AI chat service, based on the conventions in the
reference repo (`Users/davrom/code/arange/cv-management-platform/main/backend`) but stripped of
everything Azure-specific (Entra JWT, Azurite blob storage, App Insights, the
LangGraph checkpointer) since this is a generic chatbot service.

## Scope of this step

- General Python project scaffolding under `chatbot_demo/backend`.
- A `docker-compose.yml` that runs the backend + a PostgreSQL service.
- **No domain endpoints yet** — only a `/ping` health check.
- Tooling: `uv` for project/package management, `FastAPI`, `langchain`.

## Decisions locked in

- **LLM provider:** OpenAI via `langchain-openai` (`ChatOpenAI`, `OPENAI_API_KEY`).
  Matches the case's example (“t.ex. OpenAI”) and is simplest to demo. The
  factory is written so a provider registry can be added later.
- **Auth / user identity:** lightweight `X-User-Id` header dependency
  (`CurrentUserDep`) so conversations can be scoped per user. This is the seam
  where real auth slots in later, and what enables the case's privacy
  requirement (users must not see each other's conversations).

## Open questions (defaults assumed)

1. **Python version** — default assumed: `3.12` (broad image availability).
   Alternative: match the reference repo's `3.14`.
2. **Target path** — `/Users/davrom/code/chatbot_demo/` (sibling to `arange/`).

## Package layout

```
chatbot_demo/backend/
├── pyproject.toml              # uv-managed; fastapi, langchain, langchain-openai,
│                               #   sqlalchemy[asyncio], asyncpg, alembic, pydantic-settings,
│                               #   uvicorn, psycopg[binary] (alembic); dev: pytest, ruff, mypy
├── uv.lock                     # generated via `uv lock`
├── .python-version
├── .env.example                # POSTGRES_*, OPENAI_API_KEY, CHAT_MODEL, LOG_LEVEL
├── .gitignore
├── README.md                   # placeholder; the "motivering" doc grows here later
├── Dockerfile                  # 3-stage: deps → builder → slim runtime (uv), from reference
├── alembic.ini
├── alembic/
│   ├── env.py                  # async, targets Base.metadata (no LangGraph table filter)
│   ├── script.py.mako
│   └── versions/               # empty for now
├── src/chat_service/
│   ├── __init__.py
│   ├── py.typed
│   ├── config.py               # DatabaseConfig + Config (pydantic-settings, .env)
│   ├── asgi.py                 # FastAPI app, lifespan (engine dispose), /ping, timing
│   │                           #   middleware + validation handler. No domain routers yet.
│   ├── api/
│   │   └── __init__.py         # empty APIRouter aggregator, ready for v1 routers
│   ├── auth/
│   │   ├── __init__.py
│   │   └── dependencies.py     # CurrentUserDep -> reads X-User-Id header (stub)
│   ├── db/
│   │   ├── __init__.py         # exports Base
│   │   ├── models.py           # Base + current_utc() helper only (no tables yet)
│   │   └── session.py          # async engine, sessionmaker, DBSessionDep
│   └── llm/
│       ├── __init__.py
│       └── client.py           # get_chat_model() factory -> ChatOpenAI from config
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_config.py          # smoke test that config loads
```

## Root of `chatbot_demo/`

```
chatbot_demo/
├── docker-compose.yml          # backend (uv hot-reload, bind mount, named venv volume)
│                               #   + db (postgres:17, healthcheck, pgdata volume)
└── .env.example                # compose-level env (mirrors backend, POSTGRES_HOST=db)
```

## Key adaptations from the reference

- **`config.py`** keeps the `DatabaseConfig` / `Config` split (so Alembic doesn't
  need the OpenAI key) and the lazy `CONFIG` PEP-562 hook, but drops all
  Azure/Entra fields. Adds `openai_api_key`, `chat_model`
  (default e.g. `gpt-4o-mini`), `log_level`.
- **`asgi.py`** keeps the timing middleware, validation handler, and `/ping`,
  but the lifespan only disposes the DB engine — no checkpointer/blob/telemetry.
- **`llm/client.py`** is new: a small `get_chat_model()` returning a configured
  `ChatOpenAI`, mirroring the reference's `get_model()` factory pattern so it's
  easy to extend to a provider registry later.
- **`db/models.py`** ships just `Base` + `current_utc()` so the first
  migration/conversation models land cleanly in the next step.
- **`auth/dependencies.py`** provides `CurrentUserDep` resolving an `X-User-Id`
  header into a small `User` object — the seam where real auth slots in, and
  what later lets endpoints scope conversations per user (the case's privacy
  requirement).

## Verification

- `uv lock` + `uv sync` to confirm the project resolves.
- `docker compose config` to confirm the compose file is valid.
- A full `docker compose up` build is optional (pulls images and builds, takes a
  while) — run on request.

## Out of scope for this step (intentionally)

- Conversation / message DB models.
- The chat endpoints (create conversation, send message, fetch history).
- The LLM call + persistence logic.
- The written "motivering" (rationale) documentation.

These come next once the skeleton is approved.

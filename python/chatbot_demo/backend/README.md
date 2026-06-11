# chat-service

Backend for the AI chat service: FastAPI + LangChain (an OpenAI‑compatible model)
on PostgreSQL, managed with [uv](https://docs.astral.sh/uv/).

It provides API‑key auth, a streaming chat endpoint, and per‑user conversations
persisted via a LangGraph Postgres checkpointer. For the big‑picture overview,
configuration, and design rationale, see the [top‑level README](../README.md);
this file covers backend layout and local workflows.

## Layout

```
backend/
├── src/chat_service/
│   ├── config.py            # pydantic-settings: DatabaseConfig + runtime Config (.env)
│   ├── asgi.py              # FastAPI app, lifespan (opens the checkpointer), /ping
│   ├── api/
│   │   ├── chat.py          # POST /chat — NDJSON streaming, per-user session resolve
│   │   ├── conversations.py # GET /conversations + /{id}/messages, ownership checks
│   │   └── users.py         # POST /users — registration, returns an API key
│   ├── auth/                # API-key Bearer auth: get_current_user, key hashing
│   ├── agent/               # build_agent() + the checkpointer dependency
│   ├── db/                  # Base, models (Conversation, UserAccount), async session
│   └── llm/                 # get_chat_model() factory -> ChatOpenAI
├── alembic/                 # async migrations (env.py ignores checkpointer tables)
├── entrypoint.sh            # container entrypoint: alembic upgrade head, then serve
└── tests/
```

## Local development

Needs a PostgreSQL to point at — easiest is the bundled one, from the repo root:

```bash
docker compose up -d db
```

Then, from `backend/`:

```bash
cp .env.example .env          # set OPENAI_API_KEY, POSTGRES_PASSWORD; POSTGRES_HOST=localhost
uv sync                       # resolve + install dependencies
uv run alembic upgrade head   # create the conversation/user_account tables
uv run uvicorn chat_service.asgi:app --reload
```

`GET /ping` returns `{"message": "pong"}`. Interactive API docs: http://localhost:8000/docs.

Quick smoke test:

```bash
# register -> get an API key
curl -s -X POST localhost:8000/api/v1/users -H 'content-type: application/json' \
  -d '{"email":"me@example.com"}'

# chat (streams NDJSON); use the api_key from the previous response
curl -N -X POST localhost:8000/api/v1/chat -H "Authorization: Bearer <api_key>" \
  -H 'content-type: application/json' -d '{"question":"Say hi in three words."}'
```

## Tests & checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## Migrations

Models live in [src/chat_service/db/models.py](src/chat_service/db/models.py).
After changing them:

```bash
uv run alembic revision --autogenerate -m "message"
uv run alembic upgrade head
```

The LangGraph checkpointer creates and owns its own tables (`checkpoints`, …) at
runtime; Alembic's `env.py` filters them out of autogenerate so it never tries to
manage them.

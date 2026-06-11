# chat-service

Backend for a simple AI chat service: FastAPI + LangChain (OpenAI) on top of
PostgreSQL, managed with [uv](https://docs.astral.sh/uv/).

This is the scaffolding step — only a `/ping` health check exists so far.
Conversation models and the chat endpoints land in the next step.

## Layout

```
backend/
├── src/chat_service/
│   ├── config.py      # pydantic-settings Config (.env)
│   ├── asgi.py        # FastAPI app, lifespan, /ping, timing middleware
│   ├── api/           # APIRouter aggregator (no v1 routers yet)
│   ├── auth/          # CurrentUserDep — X-User-Id header (auth seam)
│   ├── db/            # Base, async engine, DBSessionDep
│   └── llm/           # get_chat_model() factory -> ChatOpenAI
├── alembic/           # async migrations targeting Base.metadata
└── tests/
```

## Local development

```bash
cp .env.example .env       # then fill in OPENAI_API_KEY and POSTGRES_PASSWORD
uv sync                    # resolve + install dependencies
uv run uvicorn chat_service.asgi:app --reload
```

`GET /ping` returns `{"message": "pong"}`.

### With docker-compose

From the repository root (`chatbot_demo/`):

```bash
cp .env.example .env       # compose-level env (POSTGRES_HOST=db)
docker compose up --build
```

This runs the backend (hot-reload) and a PostgreSQL service.

## Migrations

```bash
uv run alembic revision --autogenerate -m "message"
uv run alembic upgrade head
```

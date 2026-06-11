# Stage 1: Install Python dependencies (cached unless pyproject.toml/uv.lock change)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-editable --no-install-project

# Stage 2: Build the application (re-runs on source changes, but deps are cached)
FROM deps AS builder

COPY src/ src/
RUN uv sync --frozen --no-editable

# Stage 3: Final runtime image
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv .venv/

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "chat_service.asgi:app", "--host", "0.0.0.0", "--port", "8000"]

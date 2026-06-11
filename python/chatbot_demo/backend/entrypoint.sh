#!/usr/bin/env sh
# Backend container entrypoint: bring the database schema up to date, then start
# whatever command was passed (the server).
#
# Run as the container's entrypoint so a fresh database — e.g. the first
# `docker compose up` against an empty volume — is migrated before the app serves
# requests. The db is already accepting connections here (compose waits on its
# healthcheck), and `alembic upgrade head` is idempotent, so this is a no-op on
# subsequent starts.
set -e

echo "[entrypoint] applying database migrations: alembic upgrade head"
uv run alembic upgrade head

echo "[entrypoint] starting application: $*"
exec "$@"

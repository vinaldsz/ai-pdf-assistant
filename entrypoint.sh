#!/bin/sh
set -e

if [ $# -gt 0 ]; then
    # Fly.io release_command passes args here (e.g. "alembic upgrade head")
    exec "$@"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

#!/bin/sh
set -eu

mkdir -p "$(dirname "${DATABASE_PATH:-/app/runtime/db.sqlite3}")" /app/staticfiles /app/data /app/media

if [ "${RUN_DJANGO_MIGRATIONS:-true}" = "true" ]; then
    python manage.py migrate --noinput
fi
if [ "${RUN_COLLECTSTATIC:-true}" = "true" ]; then
    python manage.py collectstatic --noinput
fi

exec "$@"

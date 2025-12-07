#!/bin/sh

# Create db directory if it doesn't exist
mkdir -p /app/db

# Apply migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

python manage.py runserver 

exec "$@"
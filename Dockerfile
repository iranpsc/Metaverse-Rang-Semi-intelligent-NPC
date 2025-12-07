# Dockerfile
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV C_FORCE_ROOT 1  # For Celery

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make entrypoint executable (for manual use if needed)
RUN chmod +x entrypoint.sh || true

# Expose the port
EXPOSE 8000

# Default command (will be overridden by docker-compose)
CMD ["/bin/sh"]
FROM python:3.12-slim-bookworm

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .
# (Optional) COPY uv.lock .

# Install dependencies into the system python environment
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY README.md .
COPY app/ app/
COPY processors/ processors/
COPY models/ models/
COPY utils/ utils/
COPY main.py .
COPY .project-root .

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

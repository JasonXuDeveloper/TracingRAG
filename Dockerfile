FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies (without dev dependencies in production)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Copy application code
COPY tracingrag ./tracingrag

# Install the package
RUN poetry install --no-interaction --no-ansi --only-root

# Expose API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "tracingrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build stage: Install dependencies with compilation tools
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies to a virtual environment
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi --no-root --only main \
    && poetry cache clear pypi --all \
    && rm -rf /root/.cache/pypoetry

# Runtime stage: Minimal image without build tools
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY README.md ./
COPY tracingrag ./tracingrag

# Install Poetry (needed only for installing the root package)
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml for root package installation
COPY pyproject.toml ./

# Install only the root package (no dependencies)
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi --only-root \
    && rm -rf /root/.local /root/.cache

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "tracingrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
# Create a dummy src so setuptools doesn't fail on package discovery
RUN mkdir -p /app/src/fraudshield && touch /app/src/fraudshield/__init__.py
RUN pip install --upgrade pip && \
    pip install --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefix=/install .[api,dashboard,production,explainability]

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY artifacts/ /app/artifacts/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/

# Environment configurations
ENV PYTHONPATH="/app/src"

RUN useradd -m -u 1000 nonroot
USER nonroot

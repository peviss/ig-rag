FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies (cached layer separate from source)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download the default embedding model so it's baked into the image
RUN uv run python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY main.py ./
COPY api/ ./api/
COPY src/ ./src/

# Default: CLI. The api service overrides this with uvicorn.
CMD ["uv", "run", "python", "main.py"]

# BeltWatch serving image (CPU). Used for both the API and the inference worker.
FROM python:3.12.15-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first (cached unless the lockfile changes).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --extra cpu --no-group dev --no-group pipeline --no-install-project

# Application code, review UI, and configuration.
COPY src ./src
COPY frontend ./frontend
COPY configs ./configs
RUN uv sync --locked --extra cpu --no-group dev --no-group pipeline --no-editable

RUN useradd --uid 10001 --create-home beltwatch \
    && mkdir -p /data /releases \
    && chown beltwatch:beltwatch /data /releases
USER beltwatch

ENV PATH=/app/.venv/bin:$PATH \
    BELTWATCH_DATA_DIR=/data \
    BELTWATCH_RELEASES_DIR=/releases \
    BELTWATCH_FRONTEND_DIR=/app/frontend

EXPOSE 8000
CMD ["uvicorn", "beltwatch.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

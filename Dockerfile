# syntax=docker/dockerfile:1
# The API image. Two stages: the locked dependencies are resolved and installed by uv in a
# builder, and only the resulting virtualenv plus what the service needs is copied into the
# runtime image, which runs as a non-root user. Only the project dependencies are installed
# (the serving set; the pipeline's tools are the `pipeline` dependency group, left out), and
# torch is the CPU build on Linux (pyproject.toml, [tool.uv.sources]).
#
# The serving bundle (build/serving: the selected index, the pinned embedding weights and
# the benchmark questions' cached responses) is baked in, so the container needs no DVC
# data and no download at start-up. Build it first: `uv run python scripts/12_serving.py
# bundle` (or `dvc repro serving_bundle`). A clone without the DVC data pulls the published
# image instead of building (docker-compose.yml).

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
# UV_HTTP_TIMEOUT: a clean build downloads ~1 GB of wheels; uv's 30 s default failed one on a
# slow connection during the Phase 7 clean-machine check.
ENV UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.source="https://github.com/adaboranyilmaz/rag-groundedness-eval" \
      org.opencontainers.image.description="RAG over financial filings that returns a groundedness score with every answer" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/state && chown app:app /app/state
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
# the bundle before the code: it changes rarely, so a code change does not rebuild (or
# re-push) its ~310 MB layer
COPY build/serving ./serving
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
COPY results/metrics/pipeline_selection.json ./results/metrics/pipeline_selection.json
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    RAG_BUNDLE_DIR=/app/serving \
    RAG_STATE_DIR=/app/state \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
USER app
# the state volume (logs, new responses, the spend ledger) takes this directory's owner
VOLUME /app/state
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=4)"]
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

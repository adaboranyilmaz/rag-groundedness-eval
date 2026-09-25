# syntax=docker/dockerfile:1
# The API image. Two stages: the locked dependencies are resolved and installed by uv in a
# builder, and only the resulting virtualenv plus the code the service needs is copied into
# the runtime image, which runs as a non-root user. Torch is the CPU build on Linux
# (pyproject.toml, [tool.uv.sources]), so the image carries no CUDA libraries.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts
COPY results/metrics/pipeline_selection.json ./results/metrics/pipeline_selection.json
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

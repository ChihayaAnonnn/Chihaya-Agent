# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: none strictly needed for pure-Python deps, but tiktoken ships
# prebuilt wheels for linux/amd64 so no build toolchain is required.

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY bus ./bus
COPY cli ./cli
COPY cron ./cron
COPY eval ./eval
COPY heartbeat ./heartbeat
COPY providers ./providers
COPY session ./session
COPY context_file_template ./context_file_template
COPY utils.py ./

RUN pip install -e .

# DASHSCOPE_API_KEY must be provided at runtime
ENV WORKSPACE_ROOT=/app/workspaces

ENTRYPOINT ["agent"]
CMD ["roleplay", "-u", "default"]

# syntax=docker/dockerfile:1
# Offline demo image. It installs the locked runtime dependencies and runs the scripted fixture
# demo. No provider key, dataset or private artifact enters the build context (.dockerignore).

ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Dependencies first, so source edits reuse this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 celltrace \
    && useradd --system --uid 10001 --gid celltrace --home-dir /app --no-create-home celltrace \
    && install --directory --owner celltrace --group celltrace /app
WORKDIR /app
COPY --from=builder --chown=celltrace:celltrace /app/.venv /app/.venv
COPY --chown=celltrace:celltrace configs ./configs
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
USER celltrace
ENTRYPOINT ["celltrace"]
CMD ["demo", "--offline", "--out", "artifacts/public/demo"]

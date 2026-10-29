# syntax=docker/dockerfile:1
# BenchLab images, one target per benchmark family (see benchlab/registry.py `family`):
#   core       the panel + app tasks, GPQA, BFCL, TTFT, τ²-bench, GAIA, BrowseComp,
#              Terminal-Bench, SWE-bench
#   code       core + LiveCodeBench, AutomationBench, Aider (its exercises run in the
#              separate `benchlab-aider` image, see compose.yaml)
#   retrieval  core + the MTEB / BEIR / own-data stack (PyTorch)
#   serving    core + guidellm
# PyTorch (retrieval, serving) is the CPU build: no GPU here, and CUDA adds ~5 GB.
# Harnesses are installed at build time by each adapter's own prepare(), with the
# versions pinned in BenchLab, into /opt/benchlab/harnesses. Keys never enter an image.
#
# Build from the project root:  docker compose -f docker/compose.yaml --profile build build

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG DOCKER_CLI_IMAGE=docker:27.5-cli

FROM ${DOCKER_CLI_IMAGE} AS dockercli

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENCHLAB_HARNESS_DIR=/opt/benchlab/harnesses \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/var/cache/uv \
    UV_TORCH_BACKEND=cpu
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
# the Docker CLI and compose plugin, for the benchmarks that start their own containers;
# they talk to Docker only through the socket proxy (DOCKER_HOST), never a mounted socket
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
RUN pip install --no-cache-dir "uv>=0.4"
WORKDIR /opt/benchlab/src
COPY pyproject.toml README.md ./
COPY benchlab ./benchlab
RUN --mount=type=cache,target=/var/cache/uv uv pip install --system . \
 && mkdir -p /opt/benchlab/harnesses
WORKDIR /opt/benchlab

FROM base AS core
# the download cache survives rebuilds: a code change re-links packages, it doesn't
# download gigabytes again
RUN --mount=type=cache,target=/var/cache/uv BENCHLAB_IMAGE_BUILD=1 benchlab prepare --family core \
 && chmod -R a+rX /opt/benchlab
CMD ["benchlab", "--help"]

FROM core AS code
RUN --mount=type=cache,target=/var/cache/uv BENCHLAB_IMAGE_BUILD=1 benchlab prepare --family code \
 && chmod -R a+rX /opt/benchlab

FROM core AS retrieval
RUN --mount=type=cache,target=/var/cache/uv BENCHLAB_IMAGE_BUILD=1 benchlab prepare --family retrieval \
 && chmod -R a+rX /opt/benchlab

FROM core AS serving
RUN --mount=type=cache,target=/var/cache/uv BENCHLAB_IMAGE_BUILD=1 benchlab prepare --family serving \
 && chmod -R a+rX /opt/benchlab

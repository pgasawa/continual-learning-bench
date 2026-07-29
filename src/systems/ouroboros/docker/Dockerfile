# clbench-ouroboros:dev — runtime image for the DOCKER live-bridge path.
# Bakes ONLY the Ouroboros python deps; the engine SOURCE is git-cloned per run
# into <runroot>/clone by the launcher and bind-mounted at /obo/repo (so the
# image never goes stale against engine commits). Reconstructed from the
# original image's `docker history` (created 2026-06-27; arm64) — rebuild on
# x86 hosts is required (no multi-arch image was published).
#
# Build (run from an Ouroboros ENGINE clone root, e.g. ouroboros-bench-src):
#   docker build -t clbench-ouroboros:dev -f <adapter>/docker/Dockerfile .
FROM python:3.13
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

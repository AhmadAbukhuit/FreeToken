#!/usr/bin/env bash
#
# Build the release wheels inside the pytorch manylinux_2_28 CUDA container, so the
# shipped .so get a glibc 2.28 floor (the same floor as torch's own cu130 wheels)
# instead of inheriting whatever glibc the build host runs.
#
# Host usage (CI runner or a dev machine with docker):
#   scripts/ci/manylinux-build.sh
#
# The script re-execs itself inside the container; everything below the
# FT_IN_CONTAINER guard runs in the container as root.
#
# Environment (host side):
#   FT_BUILDER_IMAGE   builder image (default: pytorch/manylinux2_28-builder:cuda13.0)
#   FT_CI_CACHE_DIR    persistent cache dir on the host, holds the uv binary and
#                      uv's package cache across builds (default: ~/.cache/freetoken-ci)
#   FT_OUT_DIR         host dir that receives the wheels (default: <repo>/dist)
#   FREETOKEN_BUILD_*  passed through to scripts/build-release-wheels.sh
set -euo pipefail

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

if [[ -z "${FT_IN_CONTAINER:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  IMAGE="${FT_BUILDER_IMAGE:-pytorch/manylinux2_28-builder:cuda13.0}"
  CACHE_DIR="${FT_CI_CACHE_DIR:-$HOME/.cache/freetoken-ci}"
  OUT_DIR="${FT_OUT_DIR:-$ROOT/dist}"
  mkdir -p "$CACHE_DIR" "$OUT_DIR"

  say "building in $IMAGE"
  exec docker run --rm \
    -e FT_IN_CONTAINER=1 \
    -e FT_HOST_UID="$(id -u)" \
    -e FT_HOST_GID="$(id -g)" \
    -e FREETOKEN_BUILD_NO_STAMP="${FREETOKEN_BUILD_NO_STAMP:-}" \
    -e FREETOKEN_BUILD_STRIP="${FREETOKEN_BUILD_STRIP:-}" \
    -e FREETOKEN_KERNEL_CACHE_SPECS="${FREETOKEN_KERNEL_CACHE_SPECS:-}" \
    -e FREETOKEN_KERNEL_CACHE_VERBOSE="${FREETOKEN_KERNEL_CACHE_VERBOSE:-}" \
    -v "$ROOT:/workspace" \
    -v "$CACHE_DIR:/ci-cache" \
    -v "$OUT_DIR:/ci-out" \
    -w /workspace \
    "$IMAGE" bash scripts/ci/manylinux-build.sh
fi

# ---------------- inside the container (root) ----------------

# The mounted repo belongs to the host user; git refuses to touch it from root
# without this (and the version stamp both reads and restores via git).
git config --global --add safe.directory /workspace

# Wheels and the stamp-restored version.py are written as root into host-owned
# dirs; hand them back to the host user even when the build dies mid-way.
restore_ownership() {
  chown -R "$FT_HOST_UID:$FT_HOST_GID" /ci-out 2>/dev/null || true
  chown "$FT_HOST_UID:$FT_HOST_GID" \
    /workspace/python/freetoken/version.py /workspace/.git/index 2>/dev/null || true
}
trap restore_ownership EXIT

export PATH="/ci-cache/bin:$PATH"
if [[ ! -x /ci-cache/bin/uv ]]; then
  say "installing uv into the persistent cache"
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/ci-cache/bin sh -s -- --quiet
fi
export UV_CACHE_DIR=/ci-cache/uv

# Build venv is throwaway (recreated per build from the warm uv cache) so stale
# build deps can never linger; only the cache dir persists across builds.
VENV=/tmp/build-venv
PYBIN=/opt/python/cp312-cp312/bin/python
say "creating build venv"
uv venv --quiet --python "$PYBIN" "$VENV"
# torch must come from the cu130 index (PyPI's torch is a different CUDA variant
# and would tag the kernel-cache wheel wrong); everything else is plain PyPI.
uv pip install --quiet --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 "torch>=2.11,<2.12"
uv pip install --quiet --python "$VENV/bin/python" \
  "setuptools>=77" wheel ninja "apache-tvm-ffi>=0.1.4,<0.2"

export FREETOKEN_BUILD_PYTHON="$VENV/bin/python"
export FREETOKEN_BUILD_OUT_DIR=/ci-out
# No exec: the ownership trap above must still fire after the build returns.
bash scripts/build-release-wheels.sh

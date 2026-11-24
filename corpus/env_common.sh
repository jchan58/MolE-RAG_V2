#!/usr/bin/env bash
# Shared environment for exp70-multi. Usage: `source env_common.sh <dataset>`.
# Sets per-dataset RUN_DIR (on /work, survives login-node restarts; NON-bulk state),
# a SHARED NCBI rate-limit bucket (so concurrent dataset searches respect the account
# 10/s cap when run on the same node), the shared PubChem bulk FTP dir, and the Opus
# query-gen model pool.
set -u
DS="${1:?usage: source env_common.sh <lipo|sider|hiv|tox21>}"
export EXP70_DATASET="$DS"
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the corpus/ directory (holds the pipeline code)
export CODE="$_ENV_DIR"
export MULTI_BASE="${EXP70_BASE:-$(cd "$_ENV_DIR/.." && pwd)/corpus_runs}"   # output root (override with EXP70_BASE)
export EXP70_RUN_DIR="$MULTI_BASE/$DS"
export EXP70_RL_DIR="${EXP70_RL_DIR:-/dev/shm/exp70_multi_rl}"   # shared NCBI token bucket (node-local)
export EXP70_BULK_DIR="${EXP70_BULK_DIR:-/tmp/exp70_pubchem_bulk}"   # PubChem bulk FTP cache
# Bulk (fulltext/passages/extracted) -> node-local /tmp (never the over-quota Lustre).
export EXP70_FILTER_BASE="/tmp/exp70_${DS}"
export EXP70_BULK_ROOT="/tmp/exp70_${DS}/chunks"
# Query-gen = Claude Opus (4.6 primary + 4.5 to accelerate the 54k calls, as permitted).
export EXP70_CLAUDE_MODELS="${EXP70_CLAUDE_MODELS:-us.anthropic.claude-opus-4-6-v1,us.anthropic.claude-opus-4-5-20251101-v1:0}"
export EXP70_SAVE_RAW="${EXP70_SAVE_RAW:-0}"
export PY="${PY:-python3}"
# CODE is set at the top of this file to the corpus/ directory.
# Keep native BLAS/OpenMP single-threaded: many concurrent collection processes each
# otherwise spawn 64 BLAS threads -> pthread_create "Resource temporarily unavailable".
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$EXP70_RUN_DIR/logs" "$EXP70_RL_DIR"

#!/bin/bash
set -x
VENV=${VENV:-$HOME/exp70_venv_vllm}
PY=${PY:-python3}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/tmp/pip-cache}
"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools
# Pinned known-good combo: vllm 0.11.0 brings torch 2.8.0 (cu12x) as its dependency.
pip install "vllm==0.11.0"
echo "VLLM_BUILD_EXIT=$?"
python -c "import torch,vllm; print('torch',torch.__version__,'vllm',vllm.__version__)"
echo "VLLM_BUILD_DONE"

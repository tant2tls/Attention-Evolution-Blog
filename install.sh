#!/usr/bin/env bash
set -euo pipefail

# Run on the Linux GPU node. Install the runtime only; no model downloads.
VLLM_VENV=${VLLM_VENV:-$HOME/vllm-attention}
python3 -m venv "$VLLM_VENV"
source "$VLLM_VENV/bin/activate"
python -m pip install --upgrade uv

# Latest nightly checked 2026-11-02, pinned so reruns use the same build.
VLLM_COMMIT=17e675965d54bef8a3eb77ea92c6bae195ef793c
VLLM_VERSION=0.33.0rc1.dev450+g17e675965
uv pip install "vllm==$VLLM_VERSION" \
    --torch-backend=auto --prerelease=allow \
    --index-strategy first-index \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}"

# Check the installed version and save the environment for the run manifest.
python -c 'import sys, vllm; assert vllm.__version__ == sys.argv[1]; print("vLLM:", vllm.__version__)' "$VLLM_VERSION"
uv pip check
uv pip freeze > "$VLLM_VENV/requirements.freeze.txt"
printf 'VLLM_VERSION=%s\nVLLM_COMMIT=%s\n' "$VLLM_VERSION" "$VLLM_COMMIT" > "$VLLM_VENV/vllm-pin.txt"

# Activate in your shell before launching a model:
echo "source $VLLM_VENV/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

# Run on the Linux GPU node. Install the runtime only; no model downloads.
# Prefer an explicit scratch/local path; the recorded node has a quota-full NFS home.
VLLM_VENV=${VLLM_VENV:-$PWD/.venv-attention}
python3 -m venv "$VLLM_VENV"
source "$VLLM_VENV/bin/activate"
python -m pip install --upgrade uv

# Latest nightly checked 2026-11-02, pinned so reruns use the same build.
VLLM_COMMIT=17e675965d54bef8a3eb77ea92c6bae195ef793c
VLLM_VERSION=0.33.0rc1.dev450+g17e675965
# This wheel links CUDA 13.0; auto chose incompatible cu126 on the recorded node.
TORCH_VERSION=2.14.0+cu130
TORCH_BACKEND=cu130
uv pip install "vllm==$VLLM_VERSION" "torch==$TORCH_VERSION" \
    --torch-backend="$TORCH_BACKEND" --prerelease=allow \
    --index-strategy first-index \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}"

# Check the installed version and save the environment for the run manifest.
python -c 'import sys, vllm; assert vllm.__version__ == sys.argv[1]; print("vLLM:", vllm.__version__)' "$VLLM_VERSION"
python -c 'import sys, torch; assert torch.__version__ == sys.argv[1]; assert torch.version.cuda == "13.0"; print("torch:", torch.__version__, "CUDA:", torch.version.cuda)' "$TORCH_VERSION"
# Check the node driver/forward compatibility and a CUDA smoke operation before measuring.
uv pip check
uv pip freeze > "$VLLM_VENV/requirements.freeze.txt"
printf 'VLLM_VERSION=%s\nVLLM_COMMIT=%s\nTORCH_VERSION=%s\nTORCH_BACKEND=%s\n' "$VLLM_VERSION" "$VLLM_COMMIT" "$TORCH_VERSION" "$TORCH_BACKEND" > "$VLLM_VENV/vllm-pin.txt"

# Activate in your shell before launching a model:
echo "source $VLLM_VENV/bin/activate"

# Source from the repo root: environment for the 2026-11-03 study (no secrets here).
# HF token lives in $HF_HOME/token outside Git.
export ATTN_WORK_ROOT=${ATTN_WORK_ROOT:-/root/attn-study}
export VLLM_VENV=${VLLM_VENV:-$PWD/.venv-attention}
source "$VLLM_VENV/bin/activate"
export HF_HOME=$ATTN_WORK_ROOT/hf
export HF_HUB_OFFLINE=1
export HF_XET_HIGH_PERFORMANCE=1
_cu=$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')
# JIT caches must be on node-local disk: concurrent Triton compiles on the encrypted FUSE /root
# lose files (fixes.md B3). The overlay root is local; it is lost if the container is rescheduled.
export ATTN_JIT_ROOT=${ATTN_JIT_ROOT:-/var/tmp/attn_study_jit}
_c=$ATTN_JIT_ROOT/cu$_cu
export TRITON_CACHE_DIR=$_c/triton
export TORCHINDUCTOR_CACHE_DIR=$_c/inductor
export VLLM_CACHE_ROOT=$_c/vllm
export TILELANG_CACHE_DIR=$_c/tilelang
export FLASHINFER_WORKSPACE_BASE=$_c/flashinfer
export CUTE_DSL_CACHE_DIR=$_c/cute_dsl
export VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TILELANG_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$CUTE_DSL_CACHE_DIR"
# humming caches cubins in ~/.humming with no override (experience 2.2): keep it study-owned.
mkdir -p "$_c/humming"
if [ -L "$HOME/.humming" ] && [ "$(readlink "$HOME/.humming")" != "$_c/humming" ]; then rm "$HOME/.humming"; fi
if [ ! -e "$HOME/.humming" ]; then ln -s "$_c/humming" "$HOME/.humming"; fi

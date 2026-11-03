"""Frozen study identity, model pins and common launch policy.

Single source of truth for the 2026-11-03 attention study runners. Values here follow
attention_measure_plan.md and AGENTS.md; change them only by starting a new study.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDY_ID = os.environ.get("ATTN_STUDY_ID", "2026-11-03_h100x8_vllm033dev450_cu130_ep")
STUDY_DIR = REPO / "report" / STUDY_ID
WORK_ROOT = Path(os.environ.get("ATTN_WORK_ROOT", "/root/attn-study"))
WEIGHTS_ROOT = WORK_ROOT / "weights"

VLLM_VERSION = "0.33.0rc1.dev450+g17e675965"
VLLM_COMMIT = "17e675965d54bef8a3eb77ea92c6bae195ef793c"
TORCH_VERSION = "2.14.0+cu130"

# Contexts and batches from the plan (section 4-6).
CONTEXTS = [1024, 16384, 65536, 131072]
DECODE_BATCHES = [1, 2, 4, 8]
PREFILL_CHUNK = 8192
FIRST_CHUNK_1K = 1024
DECODE_WARMUP_STEPS = 50
DECODE_MEASURED_STEPS = 50
REPEATS = 3

COMMON = dict(
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    gpu_memory_utilization=0.82,
    enable_prefix_caching=False,
    max_num_batched_tokens=8192,
    max_num_seqs=8,
    async_scheduling=False,
    max_model_len=132096,
    enable_chunked_prefill=True,
)

MODELS: dict[str, dict] = {
    "llama-31-70b": dict(
        repo="meta-llama/Llama-3.1-70B-Instruct",
        revision="1605565b47bb9346c5515c34102e054115b4f98b",
        display="Llama 3.1 70B Instruct",
        ignore=["original/*"],
        # Dense: vLLM rejects enable_expert_parallel without experts (fixes.md B2); EP is N/A.
        extra=dict(max_model_len=131072, enable_expert_parallel=False),
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"},
        expected_regions=80,
        dense=True,
    ),
    "v4-0731": dict(
        repo="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        display="DeepSeek V4 Flash 0731",
        ignore=[],
        extra=dict(compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}),
        env={},
        expected_regions=43,
        dense=False,
    ),
    "v41": dict(
        repo="deepseek-ai/DeepSeek-V4.1-Flash",
        revision="dba1be0a40aa45a94ad051997016db3960a90277",
        display="DeepSeek V4.1 Flash",
        ignore=["assets/*", "*.pdf"],
        extra=dict(language_model_only=True),
        env={},
        expected_regions=40,
        dense=False,
    ),
    "mimo-v26": dict(
        repo="XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
        revision="2479e2d0029eca9a34cc7e7f55a121925f81908e",
        display="MiMo V2.6 Flash-MOPD",
        ignore=["dflash/*", "audio_tokenizer/*", "assets/*"],
        extra=dict(trust_remote_code=True, language_model_only=True),
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"},
        expected_regions=48,
        dense=False,
    ),
    "qwen-38-fp8": dict(
        repo="Qwen/Qwen3.8-Flash-Next-FP8",
        revision="236dfdf285828023ca3bcd3f37366c58a3469b13",
        display="Qwen3.8 Flash-Next FP8",
        ignore=[],
        extra=dict(language_model_only=True),
        env={},
        expected_regions=48,
        dense=False,
    ),
    "glm-53": dict(
        repo="zai-org/GLM-5.3-Flash",
        revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        display="GLM-5.3 Flash",
        ignore=[".eval_results/*"],
        extra=dict(),
        env={},
        expected_regions=45,
        dense=False,
    ),
}

MODEL_ORDER = ["llama-31-70b", "v4-0731", "qwen-38-fp8", "mimo-v26", "v41", "glm-53"]


def weights_dir(key: str) -> Path:
    return WEIGHTS_ROOT / key


def engine_kwargs(key: str, kv_cache_dtype: str) -> dict:
    m = MODELS[key]
    kw = dict(COMMON)
    kw.update(m["extra"])
    kw.update(
        model=str(weights_dir(key)),
        tokenizer=str(weights_dir(key)),
        kv_cache_dtype=kv_cache_dtype,
        seed=0,
    )
    return kw

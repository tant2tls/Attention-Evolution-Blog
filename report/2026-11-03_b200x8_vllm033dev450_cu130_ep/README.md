# Study `2026-11-03_b200x8_vllm033dev450_cu130_ep`

Attention-state memory, decode attention time and prefill attention time for six checkpoints on one 8 × B200 180GB node, vLLM `0.33.0rc1.dev450+g17e675965` (commit `17e675965d54…`), torch `2.14.0+cu130`, driver 580.105.08. TP8, EP on for MoE models (N/A for dense Llama), `gpu_memory_utilization=0.82`, prefix caching and speculation/MTP off, 8192-token prefill budget, `max_num_seqs=8`, synchronous scheduling, FP8 KV first. Raw token-ID prompts cut from public-domain Gutenberg texts (`bench/workload.py`). Medians of three independent repetitions; min/max and raw run paths are in `summary.csv`. This is a deployment comparison: KV/state dtypes, backends and graph modes differ by model and are listed below.

## Model status

| model | config id | KV requested → resolved | EP | accepted / control sessions | status |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 70B Instruct | `llama-31-70b__kv-fp8__be3bd823ed` | fp8 → fp8 | N/A (dense model, no experts) | llama-31-70b-s1 / llama-31-70b-overhead-s1 | see completion table |
| DeepSeek V4 Flash 0731 | `v4-0731__kv-fp8__cb79b0b71f` | fp8 → fp8_ds_mlax43 | local 32 of 256 experts per rank | v4-0731-s1 / v4-0731-overhead-s1 | see completion table |
| Qwen3.8 Flash-Next FP8 | not frozen | | | | pending / not measured |
| MiMo V2.6 Flash-MOPD | not frozen | | | | pending / not measured |
| DeepSeek V4.1 Flash | not frozen | | | | pending / not measured |
| GLM-5.3 Flash | not frozen | | | | pending / not measured |

## Per-model session completion (`model_completion.csv`)

| model | session | started (UTC) | measurement completed | shutdown verified | elapsed s | valid points / expected | status | reason |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| llama-31-70b | llama-31-70b-fp8-smoke-s1-1 | 2026-11-03T08:06:54Z |  | 2026-11-03T08:13:28Z | 393.4 | 0 / 0 | failed | validation fp8: exit code 2 |
| llama-31-70b | llama-31-70b-fp8-smoke-s1-2 | 2026-11-03T08:13:28Z | 2026-11-03T08:15:18Z | 2026-11-03T08:15:24Z | 116.2 | 5 / 0 | complete | validation fp8:  |
| llama-31-70b | llama-31-70b-s1 | 2026-11-03T08:15:45Z | 2026-11-03T08:19:04Z | 2026-11-03T08:19:11Z | 205.8 | 12 / 12 | complete | accepted:  |
| llama-31-70b | llama-31-70b-overhead-s1 | 2026-11-03T08:19:11Z | 2026-11-03T08:22:16Z | 2026-11-03T08:22:23Z | 191.8 | 9 / 9 | complete | matched unprobed control:  |
| v4-0731 | v4-0731-fp8-smoke-s1-1 | 2026-11-03T08:22:30Z | 2026-11-03T08:50:17Z | 2026-11-03T08:50:24Z | 1673.3 | 5 / 0 | complete | validation fp8:  |
| v4-0731 | v4-0731-s1 | 2026-11-03T08:50:24Z | 2026-11-03T08:55:04Z | 2026-11-03T08:55:10Z | 285.8 | 16 / 16 | complete | accepted:  |
| v4-0731 | v4-0731-overhead-s1 | 2026-11-03T08:55:11Z | 2026-11-03T08:59:28Z | 2026-11-03T08:59:34Z | 263.9 | 12 / 12 | complete | matched unprobed control:  |

Validation (smoke) sessions have expected points 0; their status reflects only the validation gate.

## Decode attention, ms per step (median of 3; batch 1 / batch 8)

| context | Llama 3.1 70B Instruct | DeepSeek V4 Flash 0731 | Qwen3.8 Flash-Next FP8 | MiMo V2.6 Flash-MOPD | DeepSeek V4.1 Flash | GLM-5.3 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 3.56 / 3.84 | 3.41 / 3.63 | – / – | – / – | – / – | – / – |
| 16K | 4.15 / 4.32 | 3.58 / 3.98 | – / – | – / – | – / – | – / – |
| 64K | 5.04 / 5.63 | 3.81 / 4.18 | – / – | – / – | – / – | – / – |
| 128K | unsupported / unsupported | 3.88 / 4.29 | – / – | – / – | – / – | – / – |

## Prefill attention, ms per chunk (median of 3)

| chunk | Llama 3.1 70B Instruct | DeepSeek V4 Flash 0731 | Qwen3.8 Flash-Next FP8 | MiMo V2.6 Flash-MOPD | DeepSeek V4.1 Flash | GLM-5.3 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K first chunk (1024) | 21.0 | 81.0 | – | – | – | – |
| 8K ending at 16K | 85.9 | 85.2 | – | – | – | – |
| 8K ending at 64K | 177.0 | 96.5 | – | – | – | – |
| 8K ending at 128K | unsupported | 134.2 | – | – | – | – |

## Attention-state memory, GiB for one request, all 8 GPUs (settled decode, occupied blocks)

| context | Llama 3.1 70B Instruct | DeepSeek V4 Flash 0731 | Qwen3.8 Flash-Next FP8 | MiMo V2.6 Flash-MOPD | DeepSeek V4.1 Flash | GLM-5.3 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.166 | 0.176 | – | – | – | – |
| 16K | 2.510 | 0.624 | – | – | – | – |
| 64K | 10.010 | 2.058 | – | – | – | – |
| 128K | unsupported | 3.969 | – | – | – | – |

## Probe overhead (matched unprobed control, whole-model ms)

| model | experiment | context | batch | probed | unprobed | overhead % | attention lower bound ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| llama-31-70b | decode | 1K | 1 | 8.40 | 7.06 | 18.9 | 2.22 |
| llama-31-70b | decode | 1K | 8 | 8.85 | 7.18 | 23.2 | 2.17 |
| llama-31-70b | decode | 16K | 1 | 9.01 | 7.67 | 17.4 | 2.82 |
| llama-31-70b | decode | 16K | 8 | 9.32 | 7.66 | 21.6 | 2.66 |
| llama-31-70b | decode | 64K | 1 | 9.92 | 8.57 | 15.7 | 3.69 |
| llama-31-70b | decode | 64K | 8 | 10.61 | 8.96 | 18.5 | 3.97 |
| v4-0731 | decode | 1K | 1 | 6.03 | 5.08 | 18.8 | 2.45 |
| v4-0731 | decode | 1K | 8 | 7.19 | 6.06 | 18.7 | 2.50 |
| v4-0731 | decode | 128K | 1 | 6.50 | 5.56 | 17.1 | 2.93 |
| v4-0731 | decode | 128K | 8 | 7.87 | 6.73 | 17.0 | 3.14 |
| v4-0731 | decode | 16K | 1 | 6.20 | 5.24 | 18.4 | 2.62 |
| v4-0731 | decode | 16K | 8 | 7.53 | 6.41 | 17.3 | 2.87 |
| v4-0731 | decode | 64K | 1 | 6.39 | 5.48 | 16.6 | 2.90 |
| v4-0731 | decode | 64K | 8 | 7.76 | 6.64 | 16.8 | 3.07 |
| llama-31-70b | prefill | 1K | 1 | 40.29 | 32.91 | 22.4 |  |
| llama-31-70b | prefill | 16K | 1 | 213.70 | 213.51 | 0.1 |  |
| llama-31-70b | prefill | 64K | 1 | 303.23 | 303.41 | -0.1 |  |
| v4-0731 | prefill | 1K | 1 | 137.15 | 126.94 | 8.0 |  |
| v4-0731 | prefill | 128K | 1 | 185.95 | 184.94 | 0.5 |  |
| v4-0731 | prefill | 16K | 1 | 141.54 | 132.89 | 6.5 |  |
| v4-0731 | prefill | 64K | 1 | 147.22 | 146.49 | 0.5 |  |

## Deployment differences that limit comparisons

| model | KV / state formats actually allocated | graph mode | other deviations |
| --- | --- | --- | --- |
| Llama 3.1 70B Instruct | FullAttentionSpec:torch.uint8 x80<br>tensors: Attention:torch.uint8 x80 | FULL_AND_PIECEWISE, breakable=1 | env `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (probes need it); engine `enable_expert_parallel=False`; max_model_len 131072 (128K points unsupported) |
| DeepSeek V4 Flash 0731 | MLAAttentionSpec:torch.uint8 x62; SlidingWindowMLASpec:torch.float32 x62; SlidingWindowMLASpec:torch.uint8 x43<br>tensors: CompressorStateCache:torch.float32 x62; DeepseekV4FlashMLAAttention:torch.float32 x2; DeepseekV4FlashMLAAttention:torch.uint8 x41; DeepseekV4IndexerCache:torch.uint8 x21; DeepseekV4SWACache:torch.uint8 x43 | FULL_DECODE_ONLY, breakable=1 | engine `compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY'}` |

FP8 K/V scales are the stock 1.0 (no calibrated scales shipped); FP8 vs BF16 state is a precision difference, not an architecture property. Weights keep each checkpoint's native mixed precision.

## Missing or incomplete measurements

- llama-31-70b memory context 131072: **unsupported**
- llama-31-70b decode context 131072 batch 1: **unsupported**
- llama-31-70b decode context 131072 batch 8: **unsupported**
- llama-31-70b prefill context 131072: **unsupported**
- Qwen3.8 Flash-Next FP8: not measured yet
- MiMo V2.6 Flash-MOPD: not measured yet
- DeepSeek V4.1 Flash: not measured yet
- GLM-5.3 Flash: not measured yet


## Definitions and limits

- **Attention time**: CUDA events (`external=True`, re-recorded on every CUDA-graph replay) at entry and
  exit of each decoder layer's attention module (projections, core, sparse index/compression, recurrent
  update, cache writes; FFN/MoE excluded). Per rank the per-layer times are summed; the node value is the
  maximum over the 8 ranks. Steps where any event was stale or outside that step's model window are rejected.
- **Whole-model time**: model-forward window + logits window on each rank; node value = max over ranks.
  Attention share uses these two compatible event-based quantities from the same (probed) steps.
- **Probe overhead**: module-level events add host/graph work. The matched unprobed session (same frozen
  config, model/logits windows only) gives the overhead; `attention_lower_bound_ms` = attention − overhead.
- **Decode**: B distinct prompts of exactly L tokens; after prefill, 50 untimed pure-decode steps with exactly
  B sequences, then 50 measured consecutive pure-decode steps. Requests that finish prefill early keep decoding
  while the others prefill, so per-request context differs slightly; first/last contexts are in `decode.csv`.
- **Prefill**: the step whose 8192-token chunk ends at L (1024-token first chunk at 1K), measured eagerly.
  `executed_positions_by_layer` records the token count each attention module received.
- **Memory**: live blocks of the single request in every KV-cache group (scheduler block tables, which see
  sliding-window frees) × each layer's runtime page size, summed over layers and the 8 ranks. Shared/KV-sharing
  consumer layers own nothing (listed as consumers in `memory_layers.csv`). `logical_payload_gib` is the
  token-proportional payload inside those pages; `slab_gib` is what the shared block pool reserves for the
  request's block ids; `gpu_allocation_delta_gib` is a sanity check only (pools are preallocated).
- Labels: every CSV number is `measured` unless marked; architecture numbers are `calculated from checkpoint`.

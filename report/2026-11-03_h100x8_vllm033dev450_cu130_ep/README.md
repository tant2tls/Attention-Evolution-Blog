# Study `2026-11-03_h100x8_vllm033dev450_cu130_ep`

Attention-state memory, decode attention time and prefill attention time for six checkpoints on one 8 × H100 80GB node, vLLM `0.33.0rc1.dev450+g17e675965` (commit `17e675965d54…`), torch `2.14.0+cu130`, driver 580.173.02. TP8, EP on for MoE models (N/A for dense Llama), `gpu_memory_utilization=0.82`, prefix caching and speculation/MTP off, 8192-token prefill budget, `max_num_seqs=8`, synchronous scheduling, FP8 KV first. Raw token-ID prompts cut from public-domain Gutenberg texts (`bench/workload.py`). Medians of three independent repetitions; min/max and raw run paths are in `summary.csv`. This is a deployment comparison: KV/state dtypes, backends and graph modes differ by model and are listed below.

## Model status

| model | config id | KV requested → resolved | EP | accepted / control sessions | status |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 70B Instruct | `llama-31-70b__kv-fp8__be3bd823ed` | fp8 → fp8 | N/A (dense model, no experts) | llama-31-70b-s1 / llama-31-70b-overhead-s1 | see completion table |
| DeepSeek V4 Flash 0731 | `v4-0731__kv-fp8__cb79b0b71f` | fp8 → fp8_ds_mlax43 | local 32 of 256 experts per rank | v4-0731-s1 / v4-0731-overhead-s1 | see completion table |
| Qwen3.8 Flash-Next FP8 | `qwen-38-fp8__kv-fp8__ef86bc5db1` | fp8 → fp8 | local 64 of 512 experts per rank | qwen-38-fp8-s1 / qwen-38-fp8-overhead-s1 | see completion table |
| MiMo V2.6 Flash-MOPD | `mimo-v26__kv-auto__6997ba48db` | auto → auto | local 32 of 256 experts per rank | mimo-v26-s1 / mimo-v26-overhead-s1 | see completion table |
| DeepSeek V4.1 Flash | not frozen | | | | pending / not measured |
| GLM-5.3 Flash | not frozen | | | | pending / not measured |

## Per-model session completion (`model_completion.csv`)

| model | session | started (UTC) | measurement completed | shutdown verified | elapsed s | valid points / expected | status | reason |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| llama-31-70b | llama-fp8-smoke1 | 2026-11-03T04:34:13Z |  | 2026-11-03T04:34:55Z | 41.7 | 0 / 0 | failed | validation fp8: exit code 2 |
| llama-31-70b | llama-fp8-smoke2 | 2026-11-03T04:35:58Z |  | 2026-11-03T04:38:47Z | 168.2 | 0 / 0 | failed | validation fp8: exit code 2 |
| llama-31-70b | llama-fp8-smoke3 | 2026-11-03T04:39:08Z | 2026-11-03T04:42:59Z | 2026-11-03T04:43:04Z | 236.2 | 3 / 0 | failed | validation fp8: exit code 4 |
| llama-31-70b | llama-fp8-smoke4 | 2026-11-03T04:43:59Z | 2026-11-03T04:45:39Z | 2026-11-03T04:45:44Z | 105.2 | 5 / 0 | complete | validation fp8:  |
| llama-31-70b | llama-31-70b-s1 | 2026-11-03T04:46:08Z | 2026-11-03T04:52:08Z | 2026-11-03T04:52:14Z | 365.3 | 15 / 18 | partial | accepted: 15/18 points with 3 valid repetitions |
| llama-31-70b | llama-31-70b-overhead-s1 | 2026-11-03T04:52:14Z | 2026-11-03T04:58:03Z | 2026-11-03T04:58:09Z | 354.8 | 15 / 15 | complete | matched unprobed control:  |
| llama-31-70b | llama-31-70b-s1 | 2026-11-03T04:46:08Z | 2026-11-03T04:52:08Z | 2026-11-03T04:52:14Z | 365.3 | 18 / 18 | complete | accepted: status recount after fixes.md B5 (3 co-collected memory points x 3 reps were omitted from the session point count); supersedes the 'partial' row for the same session, which is kept |
| v4-0731 | v4-fp8-smoke1 | 2026-11-03T04:58:32Z | 2026-11-03T05:13:29Z | 2026-11-03T05:13:36Z | 903.5 | 5 / 0 | complete | validation fp8:  |
| v4-0731 | v4-0731-s1 | 2026-11-03T05:13:58Z | 2026-11-03T05:23:20Z | 2026-11-03T05:23:27Z | 568.8 | 24 / 24 | complete | accepted:  |
| v4-0731 | v4-0731-overhead-s1 | 2026-11-03T05:23:27Z | 2026-11-03T05:32:13Z | 2026-11-03T05:32:19Z | 532.4 | 20 / 20 | complete | matched unprobed control:  |
| qwen-38-fp8 | qwen-fp8-smoke1 | 2026-11-03T05:32:23Z | 2026-11-03T05:50:45Z | 2026-11-03T05:50:55Z | 1112.7 | 5 / 0 | complete | validation fp8:  |
| qwen-38-fp8 | qwen-38-fp8-s1 | 2026-11-03T05:51:08Z | 2026-11-03T05:58:09Z | 2026-11-03T05:58:18Z | 430.8 | 24 / 24 | complete | accepted:  |
| qwen-38-fp8 | qwen-38-fp8-overhead-s1 | 2026-11-03T05:58:19Z | 2026-11-03T06:04:49Z | 2026-11-03T06:04:59Z | 400.4 | 20 / 20 | complete | matched unprobed control:  |
| mimo-v26 | mimo-fp8-smoke1 | 2026-11-03T06:05:06Z |  | 2026-11-03T06:08:53Z | 227.2 | 0 / 0 | failed | validation fp8: exit code 2 |
| mimo-v26 | mimo-auto-smoke1 | 2026-11-03T06:09:16Z | 2026-11-03T06:14:07Z | 2026-11-03T06:14:14Z | 297.8 | 5 / 0 | complete | validation auto (fp8 failed, fixes.md B7):  |
| mimo-v26 | mimo-v26-s1 | 2026-11-03T06:14:29Z | 2026-11-03T06:20:55Z | 2026-11-03T06:21:03Z | 394.3 | 24 / 24 | complete | accepted:  |
| mimo-v26 | mimo-v26-overhead-s1 | 2026-11-03T06:21:03Z | 2026-11-03T06:27:09Z | 2026-11-03T06:27:16Z | 372.3 | 20 / 20 | complete | matched unprobed control:  |

Validation (smoke) sessions have expected points 0; their status reflects only the validation gate.

## Decode attention, ms per step (median of 3; batch 1 / batch 8)

| context | Llama 3.1 70B Instruct | DeepSeek V4 Flash 0731 | Qwen3.8 Flash-Next FP8 | MiMo V2.6 Flash-MOPD | DeepSeek V4.1 Flash | GLM-5.3 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 4.15 / 4.36 | 3.56 / 3.62 | 1.73 / 2.00 | 1.95 / 2.34 | – / – | – / – |
| 16K | 4.64 / 5.50 | 3.68 / 3.93 | 1.80 / 2.13 | 2.03 / 2.61 | – / – | – / – |
| 64K | 5.98 / 8.68 | 3.82 / 4.15 | 1.88 / 2.28 | 2.26 / 3.36 | – / – | – / – |
| 128K | unsupported / unsupported | 3.88 / 4.46 | 1.90 / 2.43 | 2.55 / 4.33 | – / – | – / – |

## Prefill attention, ms per chunk (median of 3)

| chunk | Llama 3.1 70B Instruct | DeepSeek V4 Flash 0731 | Qwen3.8 Flash-Next FP8 | MiMo V2.6 Flash-MOPD | DeepSeek V4.1 Flash | GLM-5.3 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K first chunk (1024) | 25.3 | 102.4 | 63.2 | 43.0 | – | – |
| 8K ending at 16K | 142.5 | 116.6 | 55.9 | 40.1 | – | – |
| 8K ending at 64K | 358.1 | 165.2 | 63.9 | 69.5 | – | – |
| 8K ending at 128K | unsupported | 239.9 | 73.9 | 109.4 | – | – |

## Attention-state memory, GiB for one request, all 8 GPUs (settled decode, occupied blocks)

| context | Llama 3.1 70B Instruct | DeepSeek V4 Flash 0731 | Qwen3.8 Flash-Next FP8 | MiMo V2.6 Flash-MOPD | DeepSeek V4.1 Flash | GLM-5.3 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 0.166 | 0.176 | 0.189 | 0.073 | – | – |
| 16K | 2.510 | 0.624 | 0.956 | 0.733 | – | – |
| 64K | 10.010 | 2.058 | 3.499 | 2.842 | – | – |
| 128K | unsupported | 3.969 | 6.891 | 5.655 | – | – |

## Probe overhead (matched unprobed control, whole-model ms)

| model | experiment | context | batch | probed | unprobed | overhead % | attention lower bound ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| llama-31-70b | decode | 1K | 1 | 11.03 | 10.40 | 6.1 | 3.51 |
| llama-31-70b | decode | 1K | 2 | 11.13 | 10.49 | 6.1 | 3.51 |
| llama-31-70b | decode | 1K | 4 | 11.30 | 10.67 | 6.0 | 3.59 |
| llama-31-70b | decode | 1K | 8 | 11.59 | 10.97 | 5.7 | 3.73 |
| llama-31-70b | decode | 16K | 1 | 11.54 | 10.89 | 5.9 | 4.00 |
| llama-31-70b | decode | 16K | 2 | 11.63 | 10.95 | 6.2 | 3.96 |
| llama-31-70b | decode | 16K | 4 | 11.99 | 11.34 | 5.7 | 4.24 |
| llama-31-70b | decode | 16K | 8 | 12.74 | 12.10 | 5.3 | 4.87 |
| llama-31-70b | decode | 64K | 1 | 12.89 | 12.23 | 5.4 | 5.32 |
| llama-31-70b | decode | 64K | 2 | 13.00 | 12.34 | 5.3 | 5.35 |
| llama-31-70b | decode | 64K | 4 | 13.52 | 12.87 | 5.1 | 5.79 |
| llama-31-70b | decode | 64K | 8 | 15.94 | 15.29 | 4.2 | 8.03 |
| mimo-v26 | decode | 1K | 1 | 5.60 | 5.20 | 7.7 | 1.55 |
| mimo-v26 | decode | 1K | 2 | 5.84 | 5.44 | 7.4 | 1.59 |
| mimo-v26 | decode | 1K | 4 | 6.36 | 5.97 | 6.5 | 1.64 |
| mimo-v26 | decode | 1K | 8 | 7.40 | 6.96 | 6.2 | 1.91 |
| mimo-v26 | decode | 128K | 1 | 6.19 | 5.79 | 6.9 | 2.16 |
| mimo-v26 | decode | 128K | 2 | 6.50 | 6.10 | 6.6 | 2.23 |
| mimo-v26 | decode | 128K | 4 | 7.43 | 7.04 | 5.6 | 2.68 |
| mimo-v26 | decode | 128K | 8 | 9.35 | 8.93 | 4.7 | 3.91 |
| mimo-v26 | decode | 16K | 1 | 5.69 | 5.32 | 7.0 | 1.66 |
| mimo-v26 | decode | 16K | 2 | 5.93 | 5.52 | 7.5 | 1.66 |
| mimo-v26 | decode | 16K | 4 | 6.51 | 6.10 | 6.7 | 1.76 |
| mimo-v26 | decode | 16K | 8 | 7.62 | 7.29 | 4.5 | 2.28 |
| mimo-v26 | decode | 64K | 1 | 5.90 | 5.50 | 7.1 | 1.87 |
| mimo-v26 | decode | 64K | 2 | 6.13 | 5.77 | 6.3 | 1.95 |
| mimo-v26 | decode | 64K | 4 | 6.93 | 6.56 | 5.7 | 2.20 |
| mimo-v26 | decode | 64K | 8 | 8.34 | 7.95 | 4.8 | 2.97 |
| qwen-38-fp8 | decode | 1K | 1 | 6.58 | 6.02 | 9.3 | 1.16 |
| qwen-38-fp8 | decode | 1K | 2 | 6.96 | 6.40 | 8.6 | 1.36 |
| qwen-38-fp8 | decode | 1K | 4 | 7.19 | 6.62 | 8.7 | 1.36 |
| qwen-38-fp8 | decode | 1K | 8 | 7.62 | 6.98 | 9.2 | 1.36 |
| qwen-38-fp8 | decode | 128K | 1 | 6.76 | 6.20 | 8.9 | 1.34 |
| qwen-38-fp8 | decode | 128K | 2 | 7.13 | 6.60 | 8.1 | 1.59 |
| qwen-38-fp8 | decode | 128K | 4 | 7.48 | 6.90 | 8.4 | 1.64 |
| qwen-38-fp8 | decode | 128K | 8 | 8.01 | 7.42 | 7.9 | 1.84 |
| qwen-38-fp8 | decode | 16K | 1 | 6.69 | 6.12 | 9.4 | 1.23 |
| qwen-38-fp8 | decode | 16K | 2 | 7.02 | 6.48 | 8.3 | 1.45 |
| qwen-38-fp8 | decode | 16K | 4 | 7.28 | 6.71 | 8.5 | 1.46 |
| qwen-38-fp8 | decode | 16K | 8 | 7.70 | 7.09 | 8.6 | 1.52 |
| qwen-38-fp8 | decode | 64K | 1 | 6.77 | 6.18 | 9.5 | 1.29 |
| qwen-38-fp8 | decode | 64K | 2 | 7.14 | 6.56 | 8.7 | 1.51 |
| qwen-38-fp8 | decode | 64K | 4 | 7.42 | 6.83 | 8.7 | 1.56 |
| qwen-38-fp8 | decode | 64K | 8 | 7.87 | 7.27 | 8.2 | 1.69 |
| v4-0731 | decode | 1K | 1 | 8.31 | 7.56 | 9.9 | 2.82 |
| v4-0731 | decode | 1K | 2 | 8.41 | 7.75 | 8.5 | 2.91 |
| v4-0731 | decode | 1K | 4 | 8.74 | 8.11 | 7.7 | 2.94 |
| v4-0731 | decode | 1K | 8 | 9.62 | 9.09 | 5.8 | 3.10 |
| v4-0731 | decode | 128K | 1 | 8.65 | 7.92 | 9.1 | 3.16 |
| v4-0731 | decode | 128K | 2 | 8.95 | 8.22 | 8.9 | 3.28 |
| v4-0731 | decode | 128K | 4 | 9.47 | 8.72 | 8.6 | 3.38 |
| v4-0731 | decode | 128K | 8 | 10.44 | 10.03 | 4.1 | 4.05 |
| v4-0731 | decode | 16K | 1 | 8.39 | 7.68 | 9.2 | 2.97 |
| v4-0731 | decode | 16K | 2 | 8.61 | 7.93 | 8.5 | 3.08 |
| v4-0731 | decode | 16K | 4 | 9.02 | 8.35 | 8.1 | 3.17 |
| v4-0731 | decode | 16K | 8 | 9.94 | 9.43 | 5.4 | 3.43 |
| v4-0731 | decode | 64K | 1 | 8.57 | 7.85 | 9.1 | 3.10 |
| v4-0731 | decode | 64K | 2 | 8.81 | 8.14 | 8.3 | 3.25 |
| v4-0731 | decode | 64K | 4 | 9.24 | 8.52 | 8.5 | 3.26 |
| v4-0731 | decode | 64K | 8 | 10.19 | 9.69 | 5.1 | 3.66 |
| llama-31-70b | prefill | 1K | 1 | 57.88 | 52.19 | 10.9 |  |
| llama-31-70b | prefill | 16K | 1 | 374.85 | 374.19 | 0.2 |  |
| llama-31-70b | prefill | 64K | 1 | 590.09 | 590.22 | -0.0 |  |
| mimo-v26 | prefill | 1K | 1 | 103.91 | 102.40 | 1.5 |  |
| mimo-v26 | prefill | 128K | 1 | 210.50 | 210.79 | -0.1 |  |
| mimo-v26 | prefill | 16K | 1 | 140.31 | 139.69 | 0.4 |  |
| mimo-v26 | prefill | 64K | 1 | 170.79 | 170.13 | 0.4 |  |
| qwen-38-fp8 | prefill | 1K | 1 | 126.16 | 124.66 | 1.2 |  |
| qwen-38-fp8 | prefill | 128K | 1 | 185.29 | 184.83 | 0.3 |  |
| qwen-38-fp8 | prefill | 16K | 1 | 165.70 | 165.04 | 0.4 |  |
| qwen-38-fp8 | prefill | 64K | 1 | 174.92 | 174.46 | 0.3 |  |
| v4-0731 | prefill | 1K | 1 | 190.77 | 183.43 | 4.0 |  |
| v4-0731 | prefill | 128K | 1 | 354.14 | 354.38 | -0.1 |  |
| v4-0731 | prefill | 16K | 1 | 231.07 | 230.49 | 0.3 |  |
| v4-0731 | prefill | 64K | 1 | 279.36 | 279.29 | 0.0 |  |

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

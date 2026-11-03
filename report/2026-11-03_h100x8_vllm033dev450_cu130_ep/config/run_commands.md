# Exact commands used for study `2026-11-03_h100x8_vllm033dev450_cu130_ep`

Generated from the `logs/*__lifecycle.json` records, which store each session's exact child command, start time and outcome. Every command was run from the repository root after `source bench/env.sh`, wrapped by the lifecycle manager:

```bash
python bench/session.py --model-key <K> --session-id <ID> --role <ROLE> [--expected-points N] --timeout <S> -- <child command>
```

## Setup (no GPU)

```bash
bash install.sh                                  # environment: .venv-attention (vLLM 0.33.0rc1.dev450+g17e675965, torch 2.14.0+cu130)
source bench/env.sh                              # HF_HOME, node-local JIT caches (token file kept outside Git)
cp install.sh .venv-attention/vllm-pin.txt .venv-attention/requirements.freeze.txt report/2026-11-03_h100x8_vllm033dev450_cu130_ep/config/
setsid nohup config/scripts/dl_chain.sh &        # two parallel download queues (copy of the script used)
python3 bench/checkpoint_inventory.py --model-key <K>   # for each of the six models
```

## GPU sessions in execution order

| # | started (UTC) | model | session | status | elapsed s | child command |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | 2026-11-03T04:34:13Z | llama-31-70b | llama-fp8-smoke1 | failed | 41.7 | `python bench/attn_study.py --model-key llama-31-70b --session-id llama-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke` |
| 2 | 2026-11-03T04:35:58Z | llama-31-70b | llama-fp8-smoke2 | failed | 168.2 | `python bench/attn_study.py --model-key llama-31-70b --session-id llama-fp8-smoke2 --kv-cache-dtype fp8 --mode smoke` |
| 3 | 2026-11-03T04:39:08Z | llama-31-70b | llama-fp8-smoke3 | failed | 236.2 | `python bench/attn_study.py --model-key llama-31-70b --session-id llama-fp8-smoke3 --kv-cache-dtype fp8 --mode smoke` |
| 4 | 2026-11-03T04:43:59Z | llama-31-70b | llama-fp8-smoke4 | complete | 105.2 | `python bench/attn_study.py --model-key llama-31-70b --session-id llama-fp8-smoke4 --kv-cache-dtype fp8 --mode smoke` |
| 5 | 2026-11-03T04:46:08Z | llama-31-70b | llama-31-70b-s1 | complete | 365.3 | `python bench/attn_study.py --model-key llama-31-70b --session-id llama-31-70b-s1 --kv-cache-dtype fp8 --mode study` |
| 6 | 2026-11-03T04:52:14Z | llama-31-70b | llama-31-70b-overhead-s1 | complete | 354.8 | `python bench/attn_study.py --model-key llama-31-70b --session-id llama-31-70b-overhead-s1 --kv-cache-dtype fp8 --mode overhead` |
| 7 | 2026-11-03T04:58:32Z | v4-0731 | v4-fp8-smoke1 | complete | 903.5 | `python bench/attn_study.py --model-key v4-0731 --session-id v4-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke` |
| 8 | 2026-11-03T05:13:58Z | v4-0731 | v4-0731-s1 | complete | 568.8 | `python bench/attn_study.py --model-key v4-0731 --session-id v4-0731-s1 --kv-cache-dtype fp8 --mode study` |
| 9 | 2026-11-03T05:23:27Z | v4-0731 | v4-0731-overhead-s1 | complete | 532.4 | `python bench/attn_study.py --model-key v4-0731 --session-id v4-0731-overhead-s1 --kv-cache-dtype fp8 --mode overhead` |
| 10 | 2026-11-03T05:32:23Z | qwen-38-fp8 | qwen-fp8-smoke1 | complete | 1112.7 | `python bench/attn_study.py --model-key qwen-38-fp8 --session-id qwen-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke` |
| 11 | 2026-11-03T05:51:08Z | qwen-38-fp8 | qwen-38-fp8-s1 | complete | 430.8 | `python bench/attn_study.py --model-key qwen-38-fp8 --session-id qwen-38-fp8-s1 --kv-cache-dtype fp8 --mode study` |
| 12 | 2026-11-03T05:58:19Z | qwen-38-fp8 | qwen-38-fp8-overhead-s1 | complete | 400.4 | `python bench/attn_study.py --model-key qwen-38-fp8 --session-id qwen-38-fp8-overhead-s1 --kv-cache-dtype fp8 --mode overhead` |
| 13 | 2026-11-03T06:05:06Z | mimo-v26 | mimo-fp8-smoke1 | failed | 227.2 | `python bench/attn_study.py --model-key mimo-v26 --session-id mimo-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke` |
| 14 | 2026-11-03T06:09:16Z | mimo-v26 | mimo-auto-smoke1 | complete | 297.8 | `python bench/attn_study.py --model-key mimo-v26 --session-id mimo-auto-smoke1 --kv-cache-dtype auto --mode smoke` |
| 15 | 2026-11-03T06:14:29Z | mimo-v26 | mimo-v26-s1 | complete | 394.3 | `python bench/attn_study.py --model-key mimo-v26 --session-id mimo-v26-s1 --kv-cache-dtype auto --mode study` |
| 16 | 2026-11-03T06:21:03Z | mimo-v26 | mimo-v26-overhead-s1 | complete | 372.3 | `python bench/attn_study.py --model-key mimo-v26 --session-id mimo-v26-overhead-s1 --kv-cache-dtype auto --mode overhead` |
| 17 | 2026-11-03T06:27:22Z | v41 | v41-fp8-smoke1 | complete | 559.9 | `python bench/attn_study.py --model-key v41 --session-id v41-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke` |
| 18 | 2026-11-03T06:36:57Z | v41 | v41-s1 | complete | 486.9 | `python bench/attn_study.py --model-key v41 --session-id v41-s1 --kv-cache-dtype fp8 --mode study` |
| 19 | 2026-11-03T06:45:05Z | v41 | v41-overhead-s1 | complete | 439.0 | `python bench/attn_study.py --model-key v41 --session-id v41-overhead-s1 --kv-cache-dtype fp8 --mode overhead` |
| 20 | 2026-11-03T06:52:27Z | glm-53 | glm-fp8-smoke1 | complete | 645.4 | `python bench/attn_study.py --model-key glm-53 --session-id glm-fp8-smoke1 --kv-cache-dtype fp8 --mode smoke` |
| 21 | 2026-11-03T07:03:27Z | glm-53 | glm-53-s1 | failed | 238.3 | `python bench/attn_study.py --model-key glm-53 --session-id glm-53-s1 --kv-cache-dtype fp8 --mode study` |
| 22 | 2026-11-03T07:07:25Z | glm-53 | glm-53-overhead-s1 | complete | 673.0 | `python bench/attn_study.py --model-key glm-53 --session-id glm-53-overhead-s1 --kv-cache-dtype fp8 --mode overhead` |
| 23 | 2026-11-03T07:18:50Z | glm-53 | glm-53-s2 | complete | 606.4 | `python bench/attn_study.py --model-key glm-53 --session-id glm-53-s2 --kv-cache-dtype fp8 --mode study --resume-sessions glm-53-s1` |

## Between sessions

```bash
# freeze one validated deployment per model (write-once), after its smoke gate passed
python bench/freeze_config.py --model-key llama-31-70b --validated-session llama-fp8-smoke4 --kv-cache-dtype fp8 --attempts llama-fp8-smoke1,llama-fp8-smoke2,llama-fp8-smoke3,llama-fp8-smoke4
python bench/freeze_config.py --model-key v4-0731 --validated-session v4-fp8-smoke1 --kv-cache-dtype fp8 --attempts v4-fp8-smoke1
python bench/freeze_config.py --model-key qwen-38-fp8 --validated-session qwen-fp8-smoke1 --kv-cache-dtype fp8 --attempts qwen-fp8-smoke1
python bench/freeze_config.py --model-key mimo-v26 --validated-session mimo-auto-smoke1 --kv-cache-dtype auto --attempts mimo-fp8-smoke1,mimo-auto-smoke1 --fallback-reason "fp8 failed at startup: FLASH_ATTN_DIFFKV (FA4, diff-KV + sinks) asserts 'fp8_kv_dequant requires fp16 or bf16 Q' (fixes.md B7); auto validated"
python bench/freeze_config.py --model-key v41 --validated-session v41-fp8-smoke1 --kv-cache-dtype fp8 --attempts v41-fp8-smoke1
python bench/freeze_config.py --model-key glm-53 --validated-session glm-fp8-smoke1 --kv-cache-dtype fp8 --attempts glm-fp8-smoke1

# accepted session + matched unprobed control (each model); registers sessions in config/<K>__selected.json
bench/run_accepted.sh llama-31-70b fp8 s1 18 15
bench/run_accepted.sh v4-0731 fp8 s1 24 20
bench/run_accepted.sh qwen-38-fp8 fp8 s1 24 20
bench/run_accepted.sh mimo-v26 auto s1 24 20
bench/run_accepted.sh v41 fp8 s1 24 20
bench/run_accepted.sh glm-53 fp8 s1 24 20
python bench/freeze_config.py --model-key glm-53 --add-accepted glm-53-s2   # after the resumed GLM retry (fixes.md B8)

# queued next-model validations (copies in config/scripts/after_*.sh) waited for CHAIN_DONE of the previous model

# outputs (after each model)
python3 bench/analyze.py
/root/attn-study/venv-report/bin/python bench/make_report.py   # separate venv with matplotlib
python3 bench/write_readme.py
```

## Re-running the whole study on a new node

Use the faster two-phase order in [attention_measure_plan.md §7a](../../../attention_measure_plan.md):

```bash
bash install.sh && source bench/env.sh
python bench/download_model.py --model-key <K>   # all six, two parallel queues
ATTN_STUDY_ID=<new-study-id> bench/run_queue.sh s1   # phase A validate+freeze all six, phase B accepted+control
```

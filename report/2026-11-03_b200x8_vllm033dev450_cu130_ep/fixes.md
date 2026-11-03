# Fixes — study 2026-11-03_b200x8_vllm033dev450_cu130_ep

## N1. Workload corpus missing on fresh node (Llama FP8 validation attempt 1)

- **Symptom:** `llama-31-70b-fp8-smoke-s1-1` exited rc=2 after model load with
  `FileNotFoundError: /root/attn-study/workload/pg2600.txt` (`bench/workload.py corpus_text`). Not a KV/FP8 failure.
- **Cause:** the three public-domain Gutenberg texts are fetched outside the queue; the fresh B200 node had no
  `$ATTN_WORK_ROOT/workload/`. The quickstart does not list this step.
- **Fix (08:14Z):** fetched the URLs in `bench/workload.py CORPUS` with curl; all three SHA-256 match the pinned values.
- **Handling:** the queue loop was stopped before it could fall back to `auto` (stopping only the driver and
  `run_queue.sh` shells; the running session kept ownership and shut down cleanly, GPUs back to 0 MiB).
  Attempt 2 (`llama-31-70b-fp8-smoke-s1-2`, started 08:13Z, read the corpus after the fetch) passed all gates;
  Llama was frozen manually at `fp8` with both attempts listed. Attempt 1 is kept as a failed attempt and
  counts toward nothing. Rerun needed: no.
- **General lesson:** add corpus fetch + hash check to preflight (promote to docs/agent_quickstart.md).

## N2. MiMo V2.6 FP8 KV unsupported on B200 FlashAttention backend

- **Symptom:** `mimo-v26-fp8-smoke-s1-1` (09:13Z) failed at worker init on all ranks:
  `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device.`
  (H100 study failed differently: `fp8_kv_dequant requires fp16 or bf16 Q`, B7 there.)
- **Handling:** policy path unchanged — one FP8 retry, then `auto`; backends not forced. The fallback reason is
  recorded in `config/mimo-v26__selected.json`. Both failed attempts are kept and count toward nothing.

# Bugs and fixes: study `2026-11-03_h100x8_vllm033dev450_cu130_ep`

Runtime: vLLM `0.33.0rc1.dev450+g17e675965` (commit `17e675965d54…`), torch `2.14.0+cu130`,
driver 580.173.02, 8 × H100 80GB HBM3. Each entry: symptom, affected runs, cause, fix, verification,
and whether measurements need rerunning. Original evidence is never edited; a corrected run gets a new
session ID.

## B1: non-reproducible prompt salt (caught in review, before any GPU run)

- **Symptom:** `bench/attn_study.py` derived the prompt salt for warmup/smoke prompts from
  `hash(label) % 7`. Python randomizes `str` hashes per process (`PYTHONHASHSEED`), so warmup/smoke
  prompts would differ between launches. Accepted prompts were unaffected (salt 0).
- **Affected runs:** none (fixed before the first launch).
- **Fix:** fixed table `LABEL_SALT = {"accepted": 0, "warmup": 1, "smoke": 2}`.
  At the same time, an ambiguous conditional expression in the consecutive-decode-step check was
  rewritten as an explicit `consecutive = ...` boolean (its behaviour was already correct).
- **Verification:** `python -m py_compile bench/*.py`; prompt hashes repeat across processes.
- **Rerun needed:** no.

## B2: dense Llama cannot start with `--enable-expert-parallel`

- **Symptom:** session `llama-fp8-smoke1` exited in 42 s during config validation:
  `pydantic ValidationError ... Number of experts in the model must be greater than 0 when expert
  parallelism is enabled.` ([log](logs/llama-31-70b__llama-fp8-smoke1.log),
  [lifecycle](logs/llama-31-70b__llama-fp8-smoke1__lifecycle.json)). Shutdown verified; GPUs idle.
- **Affected runs:** `llama-fp8-smoke1` only (no measurements).
- **Cause:** this vLLM build validates EP against the model's expert count; Llama 3.1 70B is dense.
  The study policy says EP is enabled for MoE models and is N/A for dense Llama.
- **Fix:** `bench/study_config.py` sets `enable_expert_parallel=False` for `llama-31-70b` only; the
  runner policy gate now requires EP on for MoE models and off (= N/A) for dense models.
  Attention remains TP8. Reported as `ep_requested=N/A (dense)`, `ep_effective=no experts`.
- **Verification:** new session `llama-fp8-smoke2` (see below in this file / model_completion.csv).
- **Rerun needed:** no measurements were taken.

## B3: Triton JIT cache race on the encrypted FUSE `/root` filesystem

- **Symptom:** session `llama-fp8-smoke2` loaded weights, then all workers failed during the first
  compile with `FileNotFoundError: .../cache/cu13/triton/<hash>/_bincount_kernel.json` followed by
  `AttributeError: 'CompiledKernel' object has no attribute 'module'`; exit code 2 after 168 s
  ([log](logs/llama-31-70b__llama-fp8-smoke2.log)). Shutdown verified; GPUs returned to 0 MiB.
- **Affected runs:** `llama-fp8-smoke2` only (no measurements).
- **Cause:** `/root` on this node is `fuse.gocryptfs`. Eight TP workers compiling the same Triton
  kernel concurrently into a cache on that filesystem lose the rename/metadata race. This matches the
  inherited Frontier lesson ("Triton kernel files missing during concurrent startup on encrypted FUSE
  home"), which I had not applied because `/root` looked like a large local disk.
- **Fix:** `bench/env.sh` now puts every JIT cache (Triton, Inductor, vLLM, TileLang, FlashInfer,
  CuTe-DSL, humming via the `~/.humming` symlink) under node-local overlay disk
  `/var/tmp/attn_study_jit/cu13`. Weights stay on `/root` (read-only use is fine; 2.5 s/shard).
- **Verification:** next session `llama-fp8-smoke3`.
- **Rerun needed:** no measurements were taken.

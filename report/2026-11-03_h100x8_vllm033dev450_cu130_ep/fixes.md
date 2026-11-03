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

## B4: vLLM appends a suffix to request ids → prefill step and memory snapshots missed

- **Symptom:** session `llama-fp8-smoke3` measured all three decode points, but both prefill points
  failed with `final chunk step not found (0)`, and the co-collected memory point failed with
  `settled snapshot missing` (0 snapshots). Session exit 4 (smoke gate), shutdown verified.
  Evidence: `raw/llama-31-70b/prefill/failed-attempts/llama-fp8-smoke3__prefill_L16384__rep0.json`
  shows the scheduler key `llama-fp8-smoke3-smoke-prefill_L16384-r0-986646c0`.
- **Cause:** `InputProcessor.assign_request_id` in this build appends a random suffix to the external
  request id. The runner looked requests up in the scheduler and in the worker step metadata by
  the external id, so it found nothing. Decode selection did not depend on ids, so decode was unaffected.
- **Fix:** `Runner.internal_id()` resolves the scheduler id right after `add_request`; prefill
  selection, drive-loop live-request tracking and memory snapshots use internal ids, which are stored in
  every raw record (`internal_request_ids`).
- **Verification:** session `llama-fp8-smoke4`.
- **Rerun needed:** smoke only; no accepted data existed.

## B5: co-collected memory points missing from the accepted-session point count

- **Symptom:** `model_completion.csv` row for `llama-31-70b-s1` says `partial, 15/18 points`, although
  all 3 memory points have 3 valid repetitions (`raw/llama-31-70b/memory/repetition-{1,2,3}/`).
- **Cause:** `attn_study.py` appended only decode/prefill results to `session_summary.json`; memory is
  co-collected from the batch-1 decode run (same snapshot phase/live length rules) and was not listed, so
  `session.py` counted 15 of 18 expected points. Measurements are unaffected.
- **Fix:** the runner now appends each memory point to the session summary. The original row is kept;
  a correction row for the same session (status `complete`, 18/18 points, 54 repetitions, reason citing
  this entry) was appended. Analysis derives completeness from raw files (`summary.csv`), not from this count.
- **Verification:** `summary.csv` shows `valid_repeats=3` for all Llama memory points; next sessions
  report memory points in their summaries.
- **Rerun needed:** no.

## B6: per-layer page sizes differ inside one KV block-table group (memory accounting)

- **Symptom (caught in analysis review before any V4 memory result was reported):** V4 0731's
  scheduler `kv_cache_config` group 2 holds 62 layers (4×-compressed MLA KV, 128×-compressed MLA KV and
  indexer keys) and its group-level spec shows one page size (8640 B). The worker per-layer specs from
  `raw/v4-0731/sessions/v4-fp8-smoke1/state_buffers.json.gz` show 37440 B (4× MLA), 1728 B (128× MLA)
  and 8640 B (indexer) per block. Multiplying live blocks by the group page size would misstate memory.
- **Cause:** the group spec is a representative for block-table bookkeeping; physical per-layer tensors are
  `[num_blocks, block_size/tokens_per_state, entry_bytes]` with layer-specific entry sizes.
- **Fix:** `bench/analyze.py` uses each layer's runtime spec (`get_kv_cache_spec` captured by the probe) for
  page size, compression and window, and the group only for the live block count. Logical payload now
  counts `block_size / tokens_per_state` stored entries per block. Llama results are unchanged
  (single uniform group; verified the regenerated `summary.csv` matches).
- **Rerun needed:** no (raw evidence unchanged; analysis only).
- **Related slab fix (same review):** `slab_gib` first summed `KVCacheTensor.size` over all tensors. In V4
  the 8 `KVCacheTensor` entries are overlapping views of one block-outer buffer (each `size` = the whole
  41.7 GiB pool), which would overstate the per-block reservation 8×. Analysis now dedupes the backing
  storages of the per-layer cache tensors (`storage_ptr`, `storage_nbytes` from the probe):
  1 storage per rank for both Llama (46.9 GiB, 327680 B/block) and V4 (41.7 GiB, 1002240 B/block).

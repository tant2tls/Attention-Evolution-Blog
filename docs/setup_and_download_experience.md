# Setup and download experience

Read [experience.md](experience.md) first for later recorded sessions and verified fixes.
The inherited suggestions below are preparation leads; they are not fresh controlled measurements.
See [GPU setup and fixes](gpu_setup_and_fixes.md) for the historical problems and runtime pin.

- Before using GPU time, inspect disk space, local filesystem, host RAM, GPU topology and driver.
  Prepare scripts/configs and offline checks first. Frontier's five checkpoints used roughly 1.44 TB
  together and V4.1 used substantial pinned host RAM; include Llama and working files in the budget.
- Use local disk for weights, environments, traces and JIT caches. Frontier's encrypted FUSE home
  made trace writes slow and caused concurrent compilation races. Keep Triton, FlashInfer, TileLang
  and vLLM caches on local disk and reuse them across model relaunches on the same frozen build.
- Use `bash install.sh` and preserve its exact runtime pin. Save the installer, `vllm-pin.txt` and
  `requirements.freeze.txt` under `report/<study-id>/config/`. Capture installation and preflight
  output under that study's `logs/`. Cache version/node inventory once before timing.
- Download the pinned checkpoint/tokenizer revision into a study-owned local directory. Check
  whether a complete matching snapshot already exists before fetching it again; validate revision,
  files and integrity. Do not re-download simply because a previous server process exited.
- Frontier used `HF_XET_HIGH_PERFORMANCE=1` during downloads. Treat this as a build-specific lead:
  confirm support in the installed Hugging Face client before using it. It is not a guaranteed speedup.
- On Frontier's node, two simultaneous downloads used the link well. Use bounded parallelism only
  during preparation, with sufficient disk/RAM and within the authorized session. Never download or
  compile another model beside accepted GPU timing. Do not download into or delete a shared HF pool.
- Download in planned model order and prepare configs/offline validation while transfers run.
  Keep download start/end, revision, destination, bytes and errors in `report/<study-id>/logs/`.
  Do not count transfer/setup duration as a model's measurement duration.
- First launch may spend minutes compiling. Read progress logs before treating it as a hang;
  retain caches and allow bounded startup time. Record any retry and its cause instead of silently
  retrying forever. Stop and verify the owned server before the next model.
- Save every completed repetition immediately under the study, then update model completion status
  after its last required point and verified shutdown. This makes a resume use valid completed work
  without rerunning it. Reuse requires identical runtime/config/workload and metric definitions.

After a session, add only lessons supported by its logs. State hardware, runtime/client version,
what changed and measured benefit if available. Keep the original incident and exact fix in
`report/<study-id>/fixes.md`, then link to it here. Do not put credentials or private host paths in docs.

Current install prerequisite: use a fresh scratch/local `VLLM_VENV`, torch 2.14.0+cu130 with this
CUDA 13 vLLM wheel, and CUDA-major-specific JIT caches. Never use quota-full NFS home for caches
or environments. Export an absolute study-owned `HF_HOME`; unset `HF_HUB_OFFLINE` for authorized
downloads, then restore offline operation for measurement. Review pinned remote code and record
its hash (MiMo requires it); copy checkpoint config/inventory into the report before any authorized
checkpoint deletion. Avoid shared-pool deletion. Re-detect node/disk/GPU availability each session.

## 2026-11-03 session (study `2026-11-03_h100x8_vllm033dev450_cu130_ep`)

Verified on one 8 × H100 80GB container, driver 580.173.02, vLLM `0.33.0rc1.dev450+g17e675965`,
torch `2.14.0+cu130`. Evidence: [study fixes](../report/2026-11-03_h100x8_vllm033dev450_cu130_ep/fixes.md),
[download logs](../report/2026-11-03_h100x8_vllm033dev450_cu130_ep/logs/).

- **Downloads:** `bench/download_model.py` (pinned revision, `local_dir` study-owned, size check per file,
  `logs/download_<key>.json`). With an HF token, `HF_XET_HIGH_PERFORMANCE=1` and two parallel queues
  (`--workers 32` each) all six checkpoints (1.51 TB) finished in ~14 min: Llama 141 GB 123 s,
  V4 0731 167 GB 134 s, Qwen FP8 186 GB 174 s, MiMo 173 GB 167 s, GLM 328 GB 311 s, V4.1 510 GB 522 s.
  Keep the token in `$HF_HOME/token` outside Git (`chmod 600`).
- **Check the filesystem type, not just free space.** Here `/root` was `fuse.gocryptfs` (1.7 TB): fine for
  read-mostly weights (~2.5 s per 4.7 GB Llama shard on first load), but concurrent Triton JIT writes from
  8 workers lost files (B3). Put all JIT caches on node-local disk (`bench/env.sh`: `/var/tmp/attn_study_jit`).
- **Reloads are fast after the first launch:** with page cache and JIT caches warm, a Llama validation
  launch took 105 s end to end versus 236 s cold; V4 0731's 269 s JIT kernel warmup ran only once.
- Charts use a separate venv (`/root/attn-study/venv-report`, matplotlib) so the frozen runtime is untouched.
- **JIT archive:** after the study the node-local JIT cache (654 MB) was archived to
  `$ATTN_WORK_ROOT/jit-archive/cu13_vllm033dev450_g17e675965.tar`; `bench/env.sh` restores it when the local
  cache is empty. First launches spent ~53 of 169 GPU-session minutes compiling. Only reuse it for the identical
  vLLM/torch build (the file name carries the build).

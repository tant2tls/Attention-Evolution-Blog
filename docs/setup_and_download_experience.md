# Setup and download experience

These are inherited workflow lessons and preparation steps, not measurements of this new study.
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

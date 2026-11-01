# GPU setup and lessons carried from Frontier

Historical sources: `reproduce/README.md`, five per-model guides, `bench/serve.sh`,
`teach_me/03_traces_and_components.md`, `teach_me/04_live_kv_memory.md` and
`.claude/skills/serving/SKILL.md`. This is a focused adaptation, not an invocation of the old serving workflow.
The source hashes are in `import_provenance.json`. No old numbers become new results.

## Prepare the actual node

Record GPU UUIDs/model/memory, interconnect topology, CPU/RAM, driver, CUDA, Python, torch, vLLM
version **and commit**, dependency freeze, disk space and filesystem type. Keep one model running at a time.
Use local disk for the environment, checkpoints, traces and compilation caches. Check host RAM as well
as HBM: Frontier V4.1 kept about 190 GB of Engram tables pinned across eight ranks; this is a historical
budget observation, not a requirement guaranteed for a new EP-disabled deployment.

The five historical checkpoints occupied roughly 1.44 TB together; Llama is additional. Prefer a
study-owned download directory and only retain checkpoints that fit the authorized disk budget.
Do not download into or delete from a shared Hugging Face pool.

## Frozen runtime installer

Run `bash install.sh` from the repository on the authorized Linux GPU node. It pins the latest
official nightly observed on 2026-11-02: **vLLM `0.33.0rc1.dev450+g17e675965`**, full commit
**`17e675965d54bef8a3eb77ea92c6bae195ef793c`**. The source of truth for executable pins is `install.sh`.
The wheel was listed at the [official nightly index](https://wheels.vllm.ai/nightly/vllm/), with an
immutable link under [this commit index](https://wheels.vllm.ai/17e675965d54bef8a3eb77ea92c6bae195ef793c/vllm/).
Installation follows the [official GPU instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

The script creates/uses `~/vllm-attention` (override with `VLLM_VENV`), installs the exact
wheel, checks its version and dependency consistency, then saves `requirements.freeze.txt`
and `vllm-pin.txt` inside the environment. Activate it in your shell after running the script.
Use a new environment for a new runtime study; do not reinstall during live measurements.
Copy those records and the installer into `report/<study-id>/config/`; save setup output in its `logs/`. Record GPU/driver/CUDA details
in node preflight. Six-model compatibility remains to be validated on the GPU node.

The historical Frontier build `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`
(`0.31.1rc1.dev260+ga98247ab4`) and older `554340f3...` / `44af287e...` remain reference environments.
Their logs and behavior are inspection leads, not proof of this pinned build's implementation.

Linux setup template, to execute only in an authorized session after the version is frozen:

```bash
export VLLM_VENV=/workspace/attention-venv
bash install.sh
source "$VLLM_VENV/bin/activate"
export HF_HOME=/workspace/attention-study/hf
export TRITON_CACHE_DIR=/tmp/attention-triton
export FLASHINFER_WORKSPACE_BASE=/tmp/attention-flashinfer
export TILELANG_CACHE_DIR=/tmp/attention-tilelang
export VLLM_CACHE_ROOT=/tmp/attention-vllm
export VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$TILELANG_CACHE_DIR" "$VLLM_CACHE_ROOT"
# Copy requirements.freeze.txt and vllm-pin.txt into report/<study-id>/config/.
```

Implement a new launch configuration: TP8, **EP disabled**, utilization **0.82**, prefix caching off,
speculation/MTP off, native precision, explicit 8192-token prefill budget. Confirm each flag in the
installed CLI and loaded configuration; set max model length to support 131072 context plus the
entire warmup/timed decode window where supported. Never silently inherit Frontier's EP/0.90 defaults.
Record text-only mode and model-specific tokenizer/parser requirements. Model pins are in
`model_architectures.md`; no weight downloads are part of this transfer.

## Failures already encountered and fixes to carry forward

| Symptom / historical problem | Fix or validity rule |
| --- | --- |
| Triton `kernel.ttir` / `kernel.ptx` missing during concurrent startup on encrypted FUSE home | Move Triton/FlashInfer caches to local disk. Also set TileLang and vLLM caches locally. |
| First load spends 10–25 minutes compiling; shared-memory broadcast warnings recur | Inspect progressing logs and allow bounded startup time. Warm all measured shapes before timing. |
| FlashInfer JIT followed by `CUDA error: invalid argument` on first 0731/GLM load | Preserve failed logs, clean up owned workers, retry once after caches exist. Persistent failure stays failed. |
| Wrong runtime selected because an environment variable was unset | Use an explicit environment executable; record version/commit once before live timing, not via repeated imports beside requests. |
| Shorter generated output than requested | Historical `vllm bench serve` default `--custom-output-len 256` overrode request lengths. Set forced length explicitly if using that client; verify server-counted tokens. |
| Streamed chunks do not equal generated tokens | Count server usage / per-step annotations, including non-text special tokens. |
| Client concurrency mistaken for engine batch | Use pure decode step records with exactly B sequences; no arrivals, prefill, completion, preemption or batch changes in the timed window. |
| MiMo state looks inflated at first output token | Window state may still contain a prefill chunk. Record immediate post-prefill and settled decode snapshots separately with actual lengths. |
| V4.1 trace captures only a small portion of decode | It spans graph segments/streams. Correlate engine steps and all streams/ranks; a main-stream annotation alone is insufficient. |
| Kernel sums presented as latency | Overlap and gaps make sums different from elapsed time. Keep component sums diagnostic; use validated GPU region events/dependencies for the requested metric. |
| First profiler step has excessive all-reduce waits | Start profiler, execute an untimed primer, then capture stable steps; compare a matched unprofiled window. Do not interpret startup waits as attention cost. |
| Changed kernels fall into `other` | Inventory names/shapes from the frozen runtime and revise classifiers before accepting component attribution. `humming` can be an attention projection or expert GEMM. |
| Total GPU allocation stays flat as requests arrive | KV pools are preallocated. Observe unique live owners/occupied blocks; allocation delta and usage-times-pool are only sanity checks. |
| One usage fraction multiplied by all heterogeneous pool bytes | This was an approximation in Frontier; it cannot establish exact per-layer ownership or this study's memory total. Instrument each group/owner. |

Use exact thinking controls when chat templates are used: DeepSeek `thinking=false`; MiMo/Qwen
`enable_thinking=false`; GLM `reasoning_effort=low` with the exception recorded. Verify the installed
templates. Raw token-ID inputs may avoid template overhead if supported consistently; record that choice.

## State accounting and timing gaps that still need implementation

- Inspect each rank's live state storage, block ownership, alias offsets and recurrent buffers.
  Deduplicate shared storage regions on that rank; sum physically replicated regions across ranks
  for node bytes. Report logical useful bytes separately from rounded occupied blocks/padding and
  reserved capacity. Consumer layers reference owners without adding the same bytes again.
- A live memory allocation delta can be zero because of reserved pools. Do not call that zero live
  memory. If exact state introspection is unavailable, retain the approximation only as diagnostic
  and mark the required ownership metric `unavailable`.
- Define attention path to include Q/K/V and output projections, core attention, sparse selection,
  compression, recurrent updates/scans and state reads/writes. Save core and preparation breakdowns
  where separable. Exclude FFN and report any fused inseparable region explicitly.
- Use GPU events/region instrumentation with dependencies joining all relevant streams; record
  per-rank regions and define the node reduction without summing ranks that execute concurrently.
  Attention fraction/throughput may use only a compatible elapsed metric. A sum of local disjoint
  regions is not automatically the distributed critical path; document the exact definition.
- Validate instrumentation overhead against matched unprofiled whole-model execution. Save event
  definitions, graph/eager mode and code version. Python hooks alone can break graph replay or miss it.
- Log context growth over the at-least-50-step post-warmup decode window. At the 128K boundary reserve
  room for generation; do not compare endpoints with unrecorded extra tokens.
- The offline layer tool infers trimmed execution from relative kernel durations. Verify actual
  positions from runtime metadata/hooks; its heuristic is not an exact position count.

## Lifecycle and evidence

The new runner must own the exact server process group, enforce timeouts and clean it up in a finally
handler and signal paths. Never locate a server to kill using a broad process-name pattern. Refuse
to launch over another user's work. After shutdown verify owned worker exit and GPU usage returns to
the recorded idle baseline. Save startup, failure, retry and shutdown records.

Keep profiler captures separate from accepted timing repetitions. An annotation like
`execute_context_0(0)_generation_8(8)` is a useful historical batch check; confirm its semantics on the
chosen build. CPU step boundaries can help diagnose multi-stream captures but are not GPU elapsed
timing. The new study needs its own runners; the copied tools do not close these gaps.

## Keep the next session faster

Store each session's complete evidence under `report/<study-id>/` following
[report rules](../report/README.md), including three independent valid repetitions per point,
per-model completion times, configuration, logs and `fixes.md` when applicable.
After a verified setup/download improvement, update [working experience](setup_and_download_experience.md)
with the version, exact steps and a link to that study's incident/log. Keep model timing results and
incident-specific details in the report; `docs/` holds reusable lessons.

# Attention Measurement Plan — Minimal, Required Experiments

**Purpose:** Guide a coding agent to collect the **smallest complete set of trustworthy numbers** needed for an accessible blog explaining how attention evolved from Llama 3.1 70B to five frontier models.

**Deliverable:** Measured data and simple charts, with all related evidence under `report/<study-id>/` ([layout and rules](report/README.md)). **Do not write or redesign the blog yet.**

**Updated execution policy (2026-11-03):** Read [working experience](docs/experience.md)
before implementation or launch. It records completed and failed historical sessions on two CUDA
environments, including BF16 Qwen; those are not the new controlled six-model comparison. The active
plan uses official **Qwen FP8**, **TP8 + EP on**, and **FP8-first KV with documented fallback**.
`index.html` remains a synthetic prototype. This update changes preparation and protocol only;
GPU execution still requires a user-provided, explicitly authorized session.

Use the four unchanged Frontier checkpoint pins plus the newly pinned official Qwen FP8 revision in the architecture document (V4 means **0731**).
The experience notes identify a Llama revision prefix (`1605565b?`); recover its full immutable
revision/repository from the cited study manifest and verify it before launch. Do not invent the full pin.
Read [AGENTS.md](AGENTS.md) for evidence and lifecycle rules.

**Scope:** Six checkpoints:
1. Llama 3.1 70B Instruct — GQA baseline.
2. DeepSeek V4 Flash.
3. DeepSeek V4.1 Flash.
4. MiMo V2.6 Flash-MOPD.
5. Qwen3.8 Flash-Next-FP8 (official mixed-precision checkpoint).
6. GLM-5.3 Flash.

The precise attention types, layer counts, parameter counts and state layouts **must be verified from the deployed checkpoints**, not copied from the earlier HTML. The HTML contains synthetic values.

## 1. The four questions our blog must answer

| Question a general AI reader will ask | Number / figure needed | How to obtain it |
|---|---|---|
| **How is this model different from Llama 3?** | Total parameters, active parameters per generated token, number of layers of each attention type | Read model configuration, weights and executed graph; **not a GPU benchmark** |
| **How much memory does attention need?** | Attention-state MiB **per layer/type** and GiB **for the entire model**, at each context length | Calculate unique tensor bytes and validate against GPU allocations |
| **How fast is attention when generating a token?** | Attention time (ms/decode step) as context grows | GPU timing with fully built attention state |
| **How fast is attention when reading a prompt?** | Attention time (ms/prefill chunk) as context grows | GPU timing for a consistent prompt chunk |

**No other experiments are required for this blog.** Do not expand the study into MoE efficiency, total serving throughput, speculative decoding, prefix caching, energy, Nsight Compute bandwidth counters, synthetic kernel ablations, or quality benchmarks.

## 2. Fixed setup — same wherever technically possible

- **Runtime:** run `bash install.sh` to install vLLM `0.33.0rc1.dev450+g17e675965`,
  commit `17e675965d54bef8a3eb77ea92c6bae195ef793c` (latest official nightly observed 2026-11-02).
  The script installs from the immutable commit index and verifies the installed version. Retain
  `vllm-pin.txt` and `requirements.freeze.txt` from the environment, plus the installer and
  GPU/driver/CUDA records collected during preflight, with the study. Validate all six models before collecting accepted results. Do not upgrade mid-study.
- **Qwen checkpoint change (2026-11-02):** use `Qwen/Qwen3.8-Flash-Next-FP8`, key
  `qwen-38-fp8`, revision `236dfdf285828023ca3bcd3f37366c58a3469b13`; pin tokenizer/remote code too.
  Its [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/236dfdf285828023ca3bcd3f37366c58a3469b13/config.json) specifies FP8 quantization with dynamic activations and 128 x 128
  weight blocks, while the text dtype remains BF16 and attention projections are excluded from
  conversion. Do not infer KV/recurrent/index dtype from the model name. Verify loaded buffers.
  This is a deployment choice, not proof of smaller attention state or a precision-controlled comparison.
  Replace the BF16 deployment rather than adding another model; the study still has six models/144 points.
- **Hardware:** one node with **8 × H100 80GB** GPUs; **TP=8**, **EP enabled (`--enable-expert-parallel`)**, `gpu_memory_utilization=0.82`. Run **one model at a time**.
- **Why EP on:** the recorded V4.1 TP8/EP-off MXFP4 expert shard fails a HUMMING
  divisibility assertion ([experience ?7 G2](docs/experience.md)). Enable EP for every model launch
  to keep the requested policy consistent; for dense Llama record that it has no MoE experts.
  Attention remains TP8. Do not describe EP on as a proved fix for GLM's unrelated open startup error.
- **CUDA environment:** the pinned vLLM wheel links CUDA 13.0. Use torch **2.14.0+cu130**
  (`--torch-backend=cu130`), verify the installed torch/CUDA build and save the full dependency
  freeze. Never combine the old cu126 and cu130 studies as controls. Check actual driver/forward
  compatibility and a CUDA smoke operation before model loading; a driver string alone is insufficient.
- **Common scheduling:** `max_num_batched_tokens=8192`, `max_num_seqs=8`,
  `async_scheduling=False`, raw token-ID prompts with recorded workload hash. Set
  `max_model_len=132096` only where supported; Llama stays at 131072 without an override.
  Its 131072-token prompt plus output cannot fit: record 128K points unsupported, not extrapolated.
- **KV dtype policy:** for every model first validate explicit `--kv-cache-dtype fp8` (Python
  `kv_cache_dtype="fp8"`). If it fails to load or execute the required smoke/measurement shapes,
  retain that failed attempt and try `--kv-cache-dtype auto`. If `auto` also fails, select a documented
  model/backend-supported runnable dtype; retain the reason and exact configuration for every attempt.
  Stop rather than retrying arbitrary combinations if no supported choice works.
  Freeze one validated dtype/backend per model before its accepted repetitions; if a later shape
  requires another choice, mark affected points and rerun that model's accepted grid under the final
  choice rather than mixing settings. A fallback is not a separate precision sweep or added model.
  FP8 may resolve to `fp8_ds_mla`; it does not imply FP8 recurrent/index/tail buffers.
  Record K/V scale source/calibration and warnings; stock scale 1.0 is not evidence of equal accuracy.
- Use a supported, recorded model revision, weight dtype, attention/KV dtype, runtime build, backend and configuration. **Do not force unsupported settings.** If a model requires a different setup, report it prominently; it is a *deployment comparison*, not a controlled architecture-only comparison.
- Disable speculative decoding/MTP and cross-request prefix-cache reuse. The latter is **not** a request to run prefix-caching experiments.
- Use identical *target token lengths* with model-specific tokenization; record **actual** token counts.
- Warm up model and kernels before measuring. Use **3 valid repetitions** per point; report the median and all raw runs.
- All timings must come from GPU-synchronized execution. Distinguish **attention-region time** from whole-model forward time. If an attention region cannot be isolated reliably, record `unavailable`; **do not estimate a fake timing**.
- Keep precision/backend differences visible beside comparisons. No unsupported “model A is inherently faster” claims.

## 3. First task: verify six architectures (no GPU performance test)

For each checkpoint, extract:

| Required field | Why the reader needs it |
|---|---|
| Total unique model parameters (B) | Model size relative to Llama 3.1 70B |
| Active parameters per generated token (B) | Compute used per token, distinct from all stored parameters |
| Layer count and count of each attention type | Why per-layer and total attention memory differ |
| Attention mechanism in simple words | Full GQA, window/global, sparse, shared/compressed, recurrent/hybrid |
| Cache/state representation and ownership | KV per layer, shared KV across layers, fixed recurrent state, auxiliary index/tail |
| KV/state dtype and head/latent dimensions | Inputs for memory calculations |

**Parameter-count rules:** Count tied/shared weights only once. Separate base-model, auxiliary/MTP, conditional/lookup-table and actually loaded weights. For a routed MoE, distinguish *total stored* from *active per token*. A lookup table is not automatically a dense GEMM. If active counts differ between prefill and decode, show both with their definitions. Do not equate active parameters with resident GPU bytes.

**Required output:** `report/<study-id>/architecture.csv` and a six-model architecture table. For every numeric field, save the config path, weight-inspection evidence, or published reference. If an active count cannot be validated, mark it `unverified`, not a fabricated exact number.

## 4. Required GPU experiment A — attention-state memory

**Why:** A model with 80 KV-owning layers is not comparable to one with recurrent layers or shared KV merely by looking at the size of one layer.

**Workloads:** Each of the six models at **1K, 16K, 64K, 128K context** (1024, 16384, 65536, 131072 tokens), **one active request**. This gives **24 model–context points**, before repetitions. If 128K is unsupported, record `unsupported` and why; do not extrapolate.

At each point, collect:

1. **Memory by unique state owner** in MiB: ordinary KV, sliding-window KV, compressed/shared KV, sparse index/tail state, and recurrent state. A shared tensor is counted **once**, even if many layers read it.
2. **Memory per layer/type** in MiB: show each layer's owned state or its shared-state reference. Report a representative per-layer value and the number of such layers; do not multiply a shared tensor by the number of consumers.
3. **Whole-model attention-state total** in GiB: sum unique live state bytes for the one request, **excluding weights and temporary activations**.
4. **Observed GPU allocation change** in GiB: compare baseline after model load versus after prefill. This is a **sanity check**, not automatically equal to logical KV bytes because of reserved blocks, allocator pools and temporary buffers.

**Simple chart:** six stacked horizontal bars at the selected context length, showing **whole-model state**. A small adjacent table displays **MiB per layer/type × number of layers**, plus shared-state entries. The context selector should update both.

**Pass check:** `sum(unique owner bytes) == whole-model state bytes`. The GPU allocation check should be explained if it differs; never silently label allocated pool capacity as live KV usage.

**Required outputs:** `report/<study-id>/memory.csv`, `report/<study-id>/memory_layers.csv`, `report/<study-id>/charts/memory.png`.

## 5. Required GPU experiment B — decode attention time

**Why:** During generation, the model must use stored information for each next token.

**Workloads:** All six models × **1K, 16K, 64K, 128K context** (1024, 16384, 65536, 131072 tokens) × **batch of 1, 2, 4 and 8 actually decoding sequences** = **96 model–context–batch points**, before repetitions. A concurrency limit of 8 is **not** proof that the GPU decoded 8 sequences in a step; log actual batch size. Record unsupported/OOM cases.

**Measure these three numbers at each point:**
- **Attention time**, ms per decode step, from GPU-synchronized attention-region timing. Include sparse index selection, cache access, and recurrent-state updates as part of attention when applicable.
- **Whole-model decode-step time**, ms per step, to give the attention result context.
- **Effective attention throughput**, actual decoded tokens per GPU step divided by attention time in seconds (tokens/s); this is an amortized rate, not single-request latency.

Use fully populated state, no prompt prefill mixed into the timed steps, and at least 50 warm post-prefill decode steps. Log actual sequence lengths and batch size. Do not sum overlapping kernel times and call the result wall time.

**Simple chart:** one line per model: **decode attention ms/step vs context length**; select batch 1/2/4/8. A second small bar chart shows attention time as a fraction of whole-model decode-step time at 64K, batch 8.

**Required outputs:** `report/<study-id>/decode.csv`, `report/<study-id>/charts/decode.png`.

## 6. Required GPU experiment C — prefill attention time

**Why:** Processing long prompts can be expensive even if the final KV state is small.

**Workloads:** All six models at **1K, 16K, 64K, 128K cumulative context**, **one request** = **24 model–context points**, before repetitions.

For fair long-context comparisons, use a **fixed 8K (8192) new-token chunk** ending at 16K, 64K or 128K, with preceding state already populated. At 1K use a separate **1K first chunk**, clearly marked; do not directly rank 1K vs 8K chunk time without noting the different work size. Record the **actual number of token positions executed** inside the model, since compressed/encoder–decoder models may replay only some positions in certain layers.

**Measure just two numbers:**
- **Attention time**, ms per chunk, including attention-specific indexing, compression, recurrent scan and state writes.
- **Whole-model prefill-chunk time**, ms per chunk, as context.

**Simple chart:** grouped bars for the **8K chunk at 16K, 64K and 128K**, six models side by side. Put the 1K first-chunk result in a separate small table. Do not present analytic FLOPs as measured GPU throughput.

**Required outputs:** `report/<study-id>/prefill.csv`, `report/<study-id>/charts/prefill.png`.

## 7. Exactly what the agent should run

| Step | Task | Expected workload points |
|---|---|---:|
| 1 | Verify architecture and parameter counts for 6 checkpoints | 6 inventories |
| 2 | Memory: 4 context lengths, 1 active sequence | 24 |
| 3 | Decode: 4 contexts × batch 1/2/4/8 | 96 |
| 4 | Prefill: 1K first chunk + three 8K long-context chunks | 24 |
| **Total GPU workload configurations** | **Memory + decode + prefill** | **144** |

Each valid GPU point gets **3 repetitions**. Reuse already valid, equivalent raw measurements **only after checking all setup and metric definitions**; otherwise run them again. These are workload configurations, not promises that all will fit on the available hardware.

## 7a. Execution order and time budget (updated from study `2026-11-03_h100x8_vllm033dev450_cu130_ep`)

The first full study took **169 GPU-session minutes** on one 8 × H100 node: validation 60 min (of which
~53 min was one-time JIT compilation on first launch), failed attempts 15 min, accepted sessions 48 min,
matched unprobed controls 46 min. Measured per model, warm (JIT cached, weights in page cache):
engine init 50–120 s, accepted session 6–10 min, control 6–11 min. Cold first launch adds compilation:
Qwen ~16 min, V4 0731 ~12 min, GLM ~8 min, V4.1 ~6 min, MiMo ~3 min, Llama ~2 min. Downloads of all six pinned
checkpoints (1.51 TB) took ~14 min with an HF token and two parallel queues. Evidence:
[model_completion.csv](report/2026-11-03_h100x8_vllm033dev450_cu130_ep/model_completion.csv).

Run a new study in this order. Each step gates the next on content, never on a file merely existing.

| # | Step | Command | Why it is faster / safer |
| --- | --- | --- | --- |
| 0 | Node preflight: GPUs idle, driver, cgroup, **filesystem type of every path**, disk, RAM | `bench/env.sh`; save `logs/preflight_node.txt` | B3 cost a launch: encrypted FUSE storage breaks concurrent JIT writes. JIT caches go to node-local disk |
| 1 | Start downloads in the background (2 queues, measurement order) and pin/record revisions | `bench/download_model.py` | Overlaps 14 min of transfer with steps 2–3 |
| 2 | Restore the JIT archive of the **same build** if present | automatic in `bench/env.sh` | Saves up to ~50 min of first-launch compilation; a different build needs a new archive |
| 3 | Offline: checkpoint inventories, workload tokenization, single-GPU CUDA-graph event test | `bench/checkpoint_inventory.py`, `bench/workload.py` | Architecture table needs no GPU; catches probe problems in seconds |
| 4 | **Phase A: validate all six** (FP8 first; one FP8 retry for intermittent errors; then `auto`), freeze each | `bench/run_queue.sh <suffix>` | Satisfies "validate six-model support before freezing accepted runs"; fallbacks (MiMo: B7) are known before any accepted data; every model's JIT cache is warm for phase B |
| 5 | **Phase B: accepted session + matched control per model, in reverse order of phase A** | same script, unattended | Last-validated checkpoints are still in page cache; one resumed retry (`--resume-sessions`) keeps valid repetitions instead of rerunning a model (GLM B8) |
| 6 | After each model: `analyze.py`, charts (separate venv), `write_readme.py`, commit/push if authorized | manual | Results are safe in Git per model; a later failure cannot lose them |
| 7 | Archive the JIT cache for the next session on persistent storage | `tar -cf $ATTN_WORK_ROOT/jit-archive/...` | Next session starts warm |

Model order for phase A: **Llama → V4 0731 → Qwen FP8 → MiMo → V4.1 → GLM**. The smallest, best-known model
validates the harness first (Llama found B2 and B4 within minutes); models with the longest cold compile come
before the largest download; GLM, historically unstable, runs last so its failures cannot block others.
Phase B therefore runs GLM first (its checkpoint is still cached) and Llama last.

Expected wall time for a rerun on the same build with the JIT archive: phase A ≈ 6 × 3–5 min, phase B ≈
6 × 13–20 min, i.e. **~2–2.5 h** unattended, versus ~3 h for the first study plus debugging. Do not shortcut
the evidence rules to go faster: keep three independent repetitions, the matched unprobed control, the
smoke gate and one model at a time. Never compile or download for another model beside accepted timing.

Known per-model settings to apply from the start (no rediscovery): Llama `enable_expert_parallel=False`
(dense, B2) and `max_model_len=131072` (128K points `unsupported`); Llama/MiMo `VLLM_USE_BREAKABLE_CUDAGRAPH=1`;
V4 0731 `cudagraph_mode=FULL_DECODE_ONLY`; MiMo `trust_remote_code` (hashes in
`raw/mimo-v26/architecture/remote_code.json`) and expect FP8 KV to fail (B7) → `auto`; GLM may need the
resumed retry (B8). All are encoded in `bench/study_config.py`.

## 8. Agent implementation and evidence rules

1. **Read `docs/experience.md` first**, then the selective local handoff: `docs/model_architectures.md`, `docs/gpu_setup_and_fixes.md`, `docs/import_provenance.json` and `bench/README.md`. Consult the original repositories only for a specific unresolved implementation detail. The experience names existing GPU-runner scripts; check whether they are present and retrieve/review the referenced implementations before use. The two currently copied trace aids alone are not six-model measurement runners. Do not import historical serving sweeps or results into this study.
2. Implement a **single run manifest** recording checkpoint revision, node/GPU IDs, vLLM/torch/CUDA versions, weight dtype, requested and resolved KV dtype/format, recurrent/index/tail dtypes, scale source, fallback attempt/reason, attention backend, TP/EP, graph mode, model length, prompt tokens, chunk size, actual decode batch, repetition and timing instrumentation. Save each attempted launch config and startup/failure log.
3. Build one small runner for each of **memory**, **decode**, and **prefill**. Instrument fused attention/recurrent regions consistently; if isolation is not possible, record missing data and the exact reason. Do not replace missing GPU measurements with analytical estimates.
4. Save all outputs in `report/<study-id>/`: raw per-model/experiment/repetition folders, CSVs, charts, `config/`, `logs/`, manifest and `summary.csv` with median/min/max, valid repetition count and raw IDs. Every supported memory/decode/prefill point needs three independent valid repetitions with fresh state/requests. Failed attempts stay separate and do not count; fewer than three valid runs means incomplete. Include unsupported/unavailable cases.
5. Validate basic arithmetic, memory ownership, warmup, actual batch sizes and chunk lengths. Record model-specific differences without disguising them.
6. Produce the **three charts** and **one architecture table** above. Label every number as `measured`, `calculated from checkpoint`, or `unverified`. No synthetic values in the final measurement outputs.
7. Save per-model completion records as described below and `fixes.md` for any bugs encountered. Add reusable setup/download/fix lessons to `docs/` with links to the study evidence. Finish with `report/<study-id>/README.md`: what ran, what did not, and which comparisons are limited by dtype/backend/runtime differences. **Stop there. Do not write blog prose.**

### Experience that the implementation must preserve

| Known issue / evidence | Requirement for the new sessions |
| --- | --- |
| Qwen FP8 weights replace the measured BF16 checkpoint | Fresh checkpoint inventory, backend/dtype validation and matched controls; do not reuse BF16 execution results |
| FP8 KV validation worked for historical Qwen/Llama, V4 already packed FP8, V4.1 used FP8 | Start all six with FP8, but revalidate the selected checkpoint and runtime; MiMo FP8 and GLM successful startup are not established by these notes |
| V4 PIECEWISE sparse-indexer capture failed (F9) | Validate `FULL_DECODE_ONLY` as the recorded workaround; record graph mode and overhead |
| Llama/MiMo probes need breakable graphs; MiMo resolved class differs from config | Verify executed class, record graph deviations, and gate on model window plus expected attention regions, not merely a report file existing |
| Graph replay can leave stale CUDA events; eager sub-region probes have substantial overhead | Use replay-valid events, reject stale/out-of-window events, measure module-level regions; keep detailed breakdowns diagnostic |
| Probe overhead changes decode timing | Run matched unprobed controls under identical EP/KV/runtime/workload settings; save overhead and measured-vs-adjusted bounds without claiming an exact corrected time |
| Window/compressor blocks are freed without workers seeing frees | Obtain live block occupancy from scheduler-side metadata; do not count every historically appended block as live |
| Physical replicas, shared aliases, compressor tails and whole-block slabs matter | Check aliases, report logical/page/slab bytes separately, count replicas, retain post-prefill and settled-decode snapshots |
| A worker RPC exception can kill the worker | Catch unsupported property reads, return probe errors as data and fail the validation gate cleanly |
| GLM still failed in both CUDA environments | Validate EP-on startup first; if unresolved, retain failure and mark unavailable. Changing the runtime requires a new environment/study and fresh controls |

Probe controls and smoke validation are measurement-validity checks, not additional architecture
or quality experiments. Expected module regions: Qwen 48, Llama 80, V4 43, V4.1 40, MiMo 48,
GLM 45 only once its executed implementation is verified. Do not claim the referenced runner scripts
exist locally until inspected. Preserve three independent valid repetitions at every supported point.

## 9. GPU execution acceptance details

- Freeze a fresh study ID and one tested runtime before the first point. Historical Frontier builds
  and results are reference material only. Verify TP8, EP enabled (effective for MoE; no expert placement for dense Llama),
  utilization 0.82, the selected KV policy and actual buffer dtypes in loaded config.
- Use exact target lengths and record per-step actual lengths. A 128K input requires additional
  supported context capacity for warmup and timed generation. Record failure if that cannot fit.
- Before decode timing, complete prefill and at least 50 untimed warm decode steps; then measure
  at least 50 stable pure decode steps per repetition. Record first/last context and engine batch
  on every accepted step. Three repetitions mean three independently rebuilt request/state trials,
  not three steps from one capture. Prefill repetitions rebuild preceding state independently too.
- Settle sliding windows before the decode-state snapshot; retain immediate post-prefill readings
  separately. Record the exact snapshot phase and actual live length instead of relabelling it 1K.
- Define attention path consistently: projections, core, index selection/compression, recurrent
  updates and state operations. Record the core separately where possible. Exclude FFN; inseparable
  fused work is disclosed and the requested isolated metric stays unavailable if necessary.
- GPU timing must join relevant streams and document per-rank timing plus the node aggregation
  rule. Use compatible elapsed definitions for attention fractions and effective throughput.
  Legacy trace kernel sums, CPU spans and request-wall times do not meet this requirement.
- Memory totals sum unique live owner regions across ranks, counting physical replicas while
  avoiding aliases on each rank. Keep logical payload, rounded occupied blocks, reserved pools
  and allocation delta separate. Usage-fraction times pool is an approximate diagnostic only.
- Supported-but-unmeasurable metrics are `unavailable`; retain the point and reason. OOM,
  unsupported and failed attempts remain visible. Never fill them with HTML formulas or zeroes.
- Report profiler/event overhead against matched unprofiled step/chunk execution. Execute an
  untimed profiler primer before diagnostic captures to avoid startup synchronization artifacts.

For each of the 144 workload configurations, the manifest records its three repetitions or explicit
failure/status. Architecture inventory adds six non-performance inventories. Co-collect memory with
another experiment only when snapshot phase, live length and all required definitions match.

### Result organization and per-model completion

Follow [report/README.md](report/README.md): `report/<study-id>/` holds every related result,
config, log, raw repetition, chart and incident guide. Store reusable working experience in
[docs/](docs/README.md), including setup and faster checkpoint download lessons with versions,
commands and evidence. Keep large external traces indexed by hash in the study's `artifacts.json`.

Append each model session to `report/<study-id>/model_completion.csv`: UTC launch start, last
measurement completion, verified shutdown, terminal finish, monotonic elapsed seconds, point/repeat
counts, status, reason, manifest and log. Use the schema and timestamp definitions in the report
README. Record failures/interruption and resumed sessions too; never overwrite an attempt.
A completed model has three valid repetitions at every supported point, documented exclusions,
saved evidence and verified shutdown. Show completion time, duration and missing points in the
study README. Download/install times are recorded separately in logs.

## 10. Required CSV fields

```text
architecture.csv:
model,revision,total_params_b,active_decode_params_b,active_prefill_params_b,
layers,attention_types_and_counts,state_types,dtype,evidence,status

memory.csv:
model,revision,context_tokens,unique_kv_gib,shared_compressed_gib,
sparse_index_tail_gib,recurrent_state_gib,total_attention_state_gib,
gpu_allocation_delta_gib,snapshot_phase,actual_live_tokens,
logical_payload_gib,occupied_block_gib,repeat,status,manifest

memory_layers.csv:
model,revision,context_tokens,layer_id,attention_type,state_owner_id,
owner_or_consumer,unique_owned_mib,shared_reference,rank,state_category,repeat,status,manifest

decode.csv:
model,revision,context_tokens,target_batch,actual_batch,
attention_ms_per_step,model_ms_per_step,effective_attention_tokens_per_s,
context_start_tokens,context_end_tokens,warmup_steps,measured_steps,timing_method,repeat,status,manifest

prefill.csv:
model,revision,context_tokens,new_chunk_tokens,
executed_positions_by_layer,attention_ms_per_chunk,model_ms_per_chunk,timing_method,
repeat,status,manifest
```

For every measurement CSV, also include `config_id,ep_requested,ep_effective,
kv_dtype_requested,kv_dtype_resolved,kv_format,state_dtype_evidence,kv_scale_source,fallback_reason`.
Use one immutable `config_id` for each selected model deployment; link to attempted configurations
and logs in the manifest. Do not summarize repetitions across different configurations.

CSV quantities use binary MiB/GiB. `unique_kv_gib` excludes the separately reported shared/compressed,
index/tail and recurrent categories; add `window_kv_gib` and `ordinary_kv_gib` as disjoint components
of `unique_kv_gib`. Whole-model totals include all physical ranks; preserve per-rank raw observations.
Summary keys include model/revision, full configuration hash, workload, metric and status. Save medians,
raw repetition IDs, sample counts and evidence labels. Do not mix calculated and measured values.
The run manifest also links raw logs, per-step/per-layer records, instrumentation version, workload
hashes and shutdown evidence. Preserve large external trace hashes and authorized retrieval notes.

## 11. Definition of done

The study is complete when:
- All six checkpoint architectures have a source-backed parameter/layer inventory.
- Every **supported** point in the 144-configuration grid has three valid raw repetitions, or an explicit unsupported/failed/unavailable status with explanation. A blocked metric does not count as a measured answer.
- Per-layer state accounting reconciles to **whole-model** attention-state memory without double-counting shared KV.
- Attention timing is genuinely measured on GPU and clearly separated from whole-model timing.
- The three simple figures and the architecture comparison table can be regenerated from saved CSVs.
- All evidence, configs, logs, figures and any bug-fix guide are in `report/<study-id>/`.
- Per-model completion timestamps/durations/statuses are saved; every supported measured point
  has three independent valid repetitions with raw runs and median/min/max summaries.
- Reusable setup/download and verified fix experience is recorded in `docs/` with evidence links.
- Every model has an FP8-first validation record, EP-on resolved settings (or dense-model N/A),
  and an explicit recorded fallback if FP8 could not run. No setting change is hidden in a summary.
- No optional experiment, new blog writing, or unrequested model optimization was performed.

**Instruction to agent:** *Keep the experiment small and the explanations simple. We are measuring the architecture tradeoffs an AI reader can understand: model size, memory, decode speed and prefill speed. Do not expand the scope.*

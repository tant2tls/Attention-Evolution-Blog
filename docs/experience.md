# Attention study: working experience, fixes and results (2026-11-01 → 2026-11-02)

Self-contained summary of everything learned while running the attention study on 8 × H100 nodes:
environment, method, per-model settings, every bug and its fix, measured run times, headline results
and open items. Detailed evidence lives in the two study folders listed at the end. Private host names and personal storage paths are replaced with placeholders for publication.
Lessons here were verified on the runs they cite. Anything unconfirmed is marked **unconfirmed**.

---

## 1. Status at a glance

| model | study | status | KV dtype | notable deviations |
| --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next (`de4b8e4d…`) | cu126 | **complete**, 24/24 points × 3 reps | BF16 (auto) | none (vLLM default is breakable graphs) |
| Llama 3.1 70B Instruct (`1605565b…`) | cu126 | **complete**, 18/18 supported points × 3 | BF16 | breakable graphs opted in; 128K **unsupported** (context limit) |
| DeepSeek V4 Flash 0731 (`7872f01b…`) | cu126 | **complete**, 24/24 × 3 | FP8 (`fp8_ds_mla`, = auto) | `cudagraph_mode=FULL_DECODE_ONLY` |
| DeepSeek V4.1 Flash (`dba1be0a…`) | cu130 | **complete**, 24/24 × 3 | FP8 (`--kv-cache-dtype fp8`) | **EP on** (EP-off unsupported) |
| MiMo V2.6 Flash-MOPD (`2479e2d0…`) | cu130 | **complete**, 24/24 × 3 | BF16 | breakable graphs opted in; `trust_remote_code` |
| GLM-5.3 Flash (`eb9eb208…`) | both | **failed**: does not start | n/a | see §7 G1/F11 |

The two studies use different runtimes, so their numbers are **not one controlled set**:
- `report/2026-11-01_h100x8_vllm033dev450/`: vLLM `0.33.0rc1.dev450+g17e675965`, torch **2.14.0+cu126**
  (accidental CUDA 12/13 mix, see §2.1).
- `report/2026-11-02_h100x8_vllm033dev450_cu130/`: same vLLM commit, torch **2.14.0+cu130**.

---

## 2. Environment

### 2.1 Install: pin the torch CUDA build (most important fix)
- The pinned vLLM nightly (`17e675965d54…`) is a **CUDA 13.0 build**: all its extensions (`_C_stable_libtorch`,
  `_moe_C`, `_flashmla_C`, `_flashkda_C`, `_deepselect_C`, `_qutlass_C`) link `libcudart.so.13`, and it
  requires `nvidia-cutlass-dsl[cu13]`. But it only pins `torch==2.14.0`, with no CUDA tag.
- The original `install.sh` used `uv … --torch-backend=auto`. uv chooses from the node **kernel driver**,
  which was **575.57.08** (`nvidia-smi` shows "CUDA Version 13.0" only via forward-compat libraries).
  CUDA 13 wheels need driver ≥ 580, so uv picked **cu126**. torch 2.14.0 exists only as cu126, cu130 and
  cu132; there is no cu128/cu129.
- Result: one process with CUDA 12 torch and CUDA 13 vLLM kernels. That directly caused F8 (humming NVRTC
  headers) and was suspected for GLM. For GLM it turned out **not** to be the cause (G1).
- **Fix (in `install.sh`):** `TORCH_BACKEND=cu130` → `--torch-backend=cu130`, plus:
  - assert `torch.version.cuda` major == 13;
  - write `TORCH_VERSION`/`TORCH_BACKEND` into `vllm-pin.txt`;
  - warn when the driver is < 580.
  - Verified on driver 575 with forward compat: torch links `libcudart.so.13`, a BF16 matmul works, and all
    models except GLM run. Install time ≈ 3 min.
- Always install into a **new** `VLLM_VENV`, never `~` (see 2.2). A new runtime means a new study ID.
  - cu126 env: `/scratch/<study-user>/envs/vllm-attention`
  - cu130 env: `/scratch/<study-user>/envs/vllm-attention-cu130`

### 2.2 Node and storage facts (RunAI containers change between sessions; always re-detect)
- **Check every session:** `hostname`, `nvidia-smi -L`, the driver version, and the cgroup CPU quota
  (`/sys/fs/cgroup/cpu.max`, 198 here; `nproc` overstates). Containers seen: `<session-container>`, all 8 × H100 80GB HBM3.
- **`$HOME` (`<NFS-home>`) is quota-full (~5 GB) NFS.** Never put venvs or caches there. The default
  `VLLM_VENV=$HOME/vllm-attention` in `install.sh` would fail mid-install with "Disk quota exceeded".
- **humming caches cubins in `~/.humming`, and no env var overrides it.** That directory was moved to
  `/scratch/<study-user>/cache/humming`, with a symlink at `~/.humming`.
- **JIT caches go on node-local `/tmp`, keyed by the env's CUDA major.** `bench/env.sh` computes it from
  torch and sets `/tmp/attn_study_<uid>_cu<major>/{triton,inductor,vllm,tilelang,flashinfer,cute_dsl}`.
  Never share cubins between cu12 and cu13 envs.
- **`HF_HOME` must be exported explicitly.** The container may inherit a broken relative value.
- `/tmp` free space fluctuates (629 GB → 3.5 TB across containers) and **is lost on reschedule**. The shared scratch volume is
  99 % full (~1 TB free), so large checkpoints (V4.1 510 GB, MiMo 173 GB) went to `/tmp/attn_study_weights/`.
- **No tmux in these containers.** Use `setsid nohup …` and own the process group (`bench/session.py`).
- **The frozen vLLM env has no matplotlib.** Run `bench/make_report.py` with another Python, and never
  install into the frozen env.
- Host RAM is 1.5 TB. Re-loading a checkpoint from page cache takes ~2.5 min instead of ~4–6 min from NFS.

### 2.3 Download notes
- `bench/download_model.py --model-key K --study-dir S` downloads a pinned revision into a study-owned dir:
  - safetensors/json/tokenizer only, `original/` skipped;
  - per-model `download_ignore` / `download_extra`;
  - verifies file sizes against the pinned revision and writes `logs/download_K.json`.
- Measured: Llama 141.1 GB in 173 s; V4.1 510.3 GB in 633 s; MiMo 173.0 GB in 202 s (drafter and audio
  tokenizer excluded). Llama is gated, and the study HF token works. V4.1 and MiMo are not gated.
- `HF_HUB_OFFLINE=1` is set by `env.sh`, so unset it for downloads.

---

## 3. Launch configuration and per-model settings

Common (`bench/study_config.py::engine_kwargs`): TP8, `gpu_memory_utilization=0.82`, prefix caching
off, speculation/MTP off, `max_num_batched_tokens=8192` (chunked prefill), `max_num_seqs=8`,
`async_scheduling=False`, `max_model_len=132096`, native weight dtype, raw token-ID prompts (no chat template).
The runner asserts the resolved config (TP, EP as configured, utilization, prefix caching, speculation,
chunk, KV dtype) and refuses to measure otherwise.

| model | extra settings | why |
| --- | --- | --- |
| Qwen3.8 | `language_model_only=True` | text-only; vLLM auto-enables breakable CUDA graphs (torch.compile off) for Qwen4Exp |
| Llama 3.1 70B | `max_model_len=131072`; env `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for probed sessions | model limit; probes need breakable graphs (Llama default is torch.compile + piecewise) |
| V4 0731 | `cudagraph_mode=FULL_DECODE_ONLY`; scheduler block probe | vLLM sparse-indexer bug in PIECEWISE capture (F9); window/tail groups free blocks |
| V4.1 | `kv_cache_dtype=fp8`, `language_model_only=True`, **`enable_expert_parallel=True`**; scheduler probe | user request; EP-off TP8 unsupported (G2) |
| MiMo V2.6 | `trust_remote_code=True`, `language_model_only=True`; env `VLLM_USE_BREAKABLE_CUDAGRAPH=1`; scheduler probe; download excludes `dflash/*`, `audio_tokenizer/*` | config class only in repo code (G3); vLLM resolves the checkpoint to `MiMoV2OmniForCausalLM` |

Deviation checks, measured with unprobed sessions:
- Llama breakable vs vLLM default mode: whole-model time within **+0.7 %**.
- MiMo study (breakable) mode vs default mode: decode −3.4 % to 0 %; 1K first prefill chunk +6.8 %.

---

## 4. Measurement method (what the bench code does)

- **Probes (`bench/attn_probe.py`, loaded as vLLM `worker_extension_cls`):**
  - Imported in every worker *before* model build, so class-level patches are in place before graph capture.
  - They wrap each attention module's `forward` with `torch.cuda.Event(enable_timing=True, external=True)`.
    Inside captured (breakable) CUDA graphs these become event nodes and give valid elapsed times on replay
    (unit-tested).
  - Each event is checked against that step's model-forward window; an event not re-recorded has a negative
    offset, and the step is rejected.
  - Per step: per-GPU sum over layers; node value = max over 8 GPUs. Whole-model step = forward + logits windows.
  - Model-window class per model: Qwen4ExpForConditionalGeneration, LlamaForCausalLM, DeepseekV4ForCausalLM,
    DeepseekV41ForCausalLM, **MiMoV2OmniForCausalLM**, Glm5NextForConditionalGeneration.
- **Probe overhead is real; always run a matched unprobed launch (`ATTN_PROBE_LAYERS=0`).**
  - Module-level probes add to decode step time: Qwen 10–20 %, Llama 4–7 %, V4 5–11 %, V4.1 5–7 %, MiMo 5–11 %.
  - Prefill 8K chunks: < 1 %.
  - Report the bracket `[measured − Δ, measured]` (`overhead.csv`, `attention_lower_bound_ms`).
  - Sub-region probes inside eager-break callables (Python re-run on every replay) added ~30 % (F4): use only for composition.
- **Decode:** B requests, distinct prompts of exactly L tokens. After prefill, 50 untimed pure-decode steps
  with exactly B sequences, then 50 measured consecutive steps (actual batch and context checked every step).
  Repetitions are independent fresh requests, run in rep-major order.
- **Prefill:** the chunk whose end reaches L (8192 tokens after L−8192 populated; 1024 first chunk at 1K).
  Prefill runs eagerly. The 1K first chunk is host-launch-bound, so it isn't comparable with 8K chunks.
- **Memory:** live blocks per request per KV group × per-layer page sizes from the runtime `kv_layout.json`.
  - Three quantities: logical bytes, page bytes, slab bytes. All groups overlay one block-outermost pool,
    and a block id reserves a whole `block_stride` slab.
  - Physical replicas are counted (Qwen KV heads 4×, MLA latents 8×). An alias check runs every step.
  - **Headline snapshot = settled decode** (live = L + ~51). Post-prefill is kept separately, because V4's FP32
    compressor tails briefly hold whole-chunk blocks (18 GiB at ≥16K).
- **Live blocks for sliding-window/compressor models (V4, V4.1, MiMo) must come from the scheduler**
  (`bench/sched_probe.py`, in-process engine core `VLLM_ENABLE_V1_MULTIPROCESSING=0`). Workers only see
  appended block ids, never frees.
- **Validation gate:** `bench/smoke_gate.py REPORT N` must pass (every step has a model window and exactly N
  attention regions) before any accepted session starts.
- **Lifecycle (`bench/session.py`):**
  - Owns the process group and refuses to launch if GPUs are busy (> 2 GB).
  - Timeout; SIGTERM, then SIGKILL to the group; waits for the group to exit and GPU memory to return to baseline.
  - Appends `model_completion.csv` and refuses reused session ids.

---

## 5. Runbook (copy-paste order for one model)

```bash
# 0. node check
hostname; nvidia-smi -L; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1; cat /sys/fs/cgroup/cpu.max
# 1. env (once per runtime)
VLLM_VENV=/scratch/<study-user>/envs/vllm-attention-cu130 bash install.sh
export VLLM_VENV=/scratch/<study-user>/envs/vllm-attention-cu130; source bench/env.sh
S=report/<study-id>; ID=$(basename $S); mkdir -p $S/{config,logs,raw}; cp install.sh $VLLM_VENV/vllm-pin.txt $VLLM_VENV/requirements.freeze.txt $S/config/
# 2. weights + inventory
(unset HF_HUB_OFFLINE; python bench/download_model.py --model-key K --study-dir $S --workers 32)
python3 bench/checkpoint_inventory.py --ckpt <path> --model-key K --out $S/raw/K/architecture
# 3. validation launch, then gate (N = expected attention regions)
ATTN_PROBE_MODEL=K ATTN_PROBE_LEVEL=attn python bench/session.py --study-dir $S --study-id $ID --model-key K \
  --revision <rev> --session-id K-smoke1 --expected-points 0 --no-completion-row -- \
  python bench/smoke.py --model-key K --out $S/logs/K-smoke1_report.json
python3 bench/smoke_gate.py $S/logs/K-smoke1_report.json N
# 4. accepted session, then matched unprobed session (add VLLM_USE_BREAKABLE_CUDAGRAPH=1 for Llama/MiMo)
ATTN_PROBE_MODEL=K ATTN_PROBE_LEVEL=attn python bench/session.py ... --session-id K-s1 --expected-points 24 -- \
  python bench/attn_study.py --model-key K --study-dir $S --session-id K-s1
ATTN_PROBE_MODEL=K ATTN_PROBE_LAYERS=0 python bench/session.py ... --session-id K-overhead-s1 --expected-points 24 -- \
  python bench/attn_study.py --model-key K --study-dir $S --session-id K-overhead-s1 --mode overhead
# 5. report (Python with matplotlib)
python3 bench/make_report.py --study-dir $S --models K1,K2
```
Expected attention regions N: Qwen 48 (36 GDN + 12 QSA), Llama 80, V4 43, V4.1 40, MiMo 48.

---

## 6. Per-model notes

- **Qwen3.8 Flash-Next:**
  - 48 layers: 36 Gated DeltaNet + 12 QSA (full K/V stored; the indexer picks 2048 from 4×-compressed keys).
  - 180.0 B params stored (51.2 B n-gram lookup table, CPU-offloaded "Engram"; 2.6 B MTP), 6.67 B dense active per token.
  - KV block 400 tokens (unified with mamba pages; mamba page padded 2.17 %).
  - The 2 KV heads are replicated 4× over TP8 (counted in memory).
- **Llama 3.1 70B:** 80 GQA layers (64 Q / 8 KV heads), 70.55 B params, 320 KiB K/V per token. `max_position_embeddings`
  = 131072: vLLM rejects a 131072-token prompt plus ≥1 output, so all 128K points are **unsupported**
  (not raised via `VLLM_ALLOW_LONG_MAX_MODEL_LEN`).
- **DeepSeek V4 0731:**
  - 43 MLA layers (128-token window; 21 add 4× compressed KV + indexer, 20 add 128× compressed, 2 window-only).
  - 304.8 B params (MXFP4 experts are 2 params per I8 byte; scales excluded), 13.27 B active.
  - `auto` KV already means FP8 (`fp8_ds_mla`, 584 B/token). The first launch's JIT warmup takes ~265 s.
- **DeepSeek V4.1:**
  - 40 layers; KV shared from source layers [2, 8, 14, 20]; 196.6 B FP8 Engram lookup table.
  - 763.6 B params stored (543.6 B MXFP4 experts), 16.1 B active.
  - Specialized decode is opt-in only (`additional_config.decode_backend`); the default stock path is what was measured.
- **MiMo V2.6 Flash-MOPD:**
  - 48 layers: 9 full (layers 0, 5, 11, …, 47) + 39 sliding-window (128 tokens, attention sinks); separate K/V dims (192/128).
  - 310.8 B params (302.8 B MXFP4 experts; `store_dtype: mxfp4`), 14.8 B active.
  - `config.json` says `MiMoV2ForCausalLM`, but vLLM resolves it to `MiMoV2OmniForCausalLM`.
- **GLM-5.3 Flash:**
  - 45 layers (34 KDA linear + 11 DSA sparse).
  - Loads (38.8 GiB/GPU) and survives memory profiling, then dies on the first small-batch forward (§7 G1/F11).
  - On paper its sparse backend (`FLASHINFER_MLA_SPARSE_SM90`) accepts `fp8` KV, but this is unverified.

### KV-cache dtype check (`--kv-cache-dtype fp8`, validation launches)
| model | runs? | effect |
| --- | --- | --- |
| Qwen3.8 | yes | only QSA K/V → FP8 (512 B/token/layer/GPU); index keys/ring stay BF16; GDN state unchanged; token slots 2.47 M → 4.67 M; 48/48 greedy tokens identical to BF16 |
| Llama | yes | K/V → FP8, same FA3 backend; slots 1.21 M → 2.41 M; greedy output diverges after 8 tokens (both coherent) |
| V4 0731 | yes, no-op | `auto` already resolves to `fp8_ds_mla`; identical layout |
| V4.1 | yes | measured with fp8 |
| GLM | unknown | model does not start |
Caveat: no calibrated K/V scales ship with Qwen/Llama, so vLLM uses scale 1.0 and warns about accuracy.

---

## 7. Bugs and fixes (symptom → cause → fix)

Study-specific ids: F = cu126 study, G = cu130 study. Ops = operational mistakes worth not repeating.

| id | symptom | cause | fix |
| --- | --- | --- | --- |
| F1 | all workers: `ImportError … partially initialized module vllm.models.qwen4_exp.nvidia.qsa` | probe imported `qsa.py` before `nvidia/model.py` (circular import) | import `vllm.models.qwen4_exp.nvidia.model` first |
| F2 | probe RPC `NotImplementedError` from `kv_cache_interface.tokens_per_state` | vLLM spec base properties raise `NotImplementedError`; `hasattr` doesn't catch it | guard every spec attribute read with `try/except` |
| F3 | `'KVCacheTensor' object has no attribute 'shared_by'`; following RPCs dead | field removed in this build (now `layers, layer_stride, block_stride, offset`); **an exception in a worker RPC kills that worker** | generic serialization; every probe RPC returns errors as data (`_safe`); hooks never raise |
| F4 | probed Qwen decode ~30 % slower than unprobed | sub-region probes inside eager-break callables run Python every graph replay | primary sessions use `ATTN_PROBE_LEVEL=attn` (module entry/exit only); full probes only for a labelled breakdown |
| F5 | (caught) a second overhead session would overwrite `repetition-N/point.json` | point dirs not namespaced by session | dirs `overhead-<session>_<exp>`, `breakdown-<session>_<exp>`; never overwrite (`collisions/`) |
| F6 | Llama s1: runner guard rejected the 128K prefill-only run | own guard counted `max_tokens` | prefill-only requires `context <= max_model_len` |
| F7 | Llama s2: `VLLMValidationError: prompt (131072) + ≥1 output > max_model_len` | real limit (F6 assumption wrong) | record 128K memory/prefill/decode as `unsupported`; runner skips contexts with `L+1 > max_model_len` |
| F8 | V4: `NVRTCCompiler failed`, `cannot open source file "cuda/std/cstdint"` (humming MXFP4 JIT) | humming chose CUDA 12 headers (torch cu126) but compiled with CUDA 13 NVRTC; CCCL is in `nvidia/cu13/include/cccl` | probe appends that dir to `humming.jit.compiler.NVRTCCompiler._get_include_dirs` when missing; the root fix is the cu130 install (§2.1) |
| F9 | V4: `ValueError: max() iterable argument is empty` in `sparse_attn_indexer.py:676` during PIECEWISE capture | vLLM indexer builder emits no prefill chunks for a 0-length dummy query | `cudagraph_mode=FULL_DECODE_ONLY` (decode graphs unchanged) |
| F10 | V4: thousands of block "duplicates"; also a missing region expectation (`KeyError`) | window/compressor groups free and reuse blocks; workers never see frees | scheduler-side block probe; add expected region counts per model |
| F11 / G1 | GLM: bare `CUDA error: invalid argument` from all workers at the first small-batch graph capture or warmup forward; no traceback even with `CUDA_LAUNCH_BLOCKING=1` | **unknown.** Not KV dtype, graph mode (incl. none), all-reduce backend (mnnvl/trtllm/off), allreduce-RMS fusion, the CUDA 12/13 mix (fails on cu130 too), or the MoE backend (TRITON and FLASHINFER_CUTLASS) | open. Remaining differences vs Frontier's working GLM: **EP on**, vLLM `0.31.1rc1.dev50+g554340f3d` with torch cu132, utilization 0.90, `max_num_seqs` 64. Suspect a GLM custom kernel on small token counts (CuTe-DSL runtime prints that bare message; **unconfirmed**) |
| G2 | V4.1 (EP off): `assert config.hidden_size % size == 0` in `humming/tune/process_input.py` | moe_intermediate 2304 / TP8 = 288-wide expert shard; MXFP4 kernel needs a power-of-two group divisor | **EP on** for V4.1 (documented deviation; attention stays TP8). V4 0731 is fine (2048/8 = 256) |
| G3 | MiMo: "repository contains custom code … pass `trust_remote_code=True`" | no native config class for `model_type: mimo_v2` | fetch `configuration_mimo_v2.py` (+ modeling) at the pinned revision, review (PretrainedConfig only, no I/O), record sha256 (`773062ac…`, `a8c3cb3a…`), `trust_remote_code=True`, `download_extra` |
| G4 | MiMo: 48 attention regions recorded but every step `model window never recorded` | probe wrapped `MiMoV2ForCausalLM`/`MiMoV2FlashForCausalLM`; vLLM runs `MiMoV2OmniForCausalLM` ("Resolved architecture" log line) | wrap the class from the engine's `Resolved architecture` line, not `config.json`; smoke gate added |

### 7.1 Additional fixes, 2026-11-03 study (same vLLM commit, cu130)

Details and evidence: [fixes.md](../report/2026-11-03_h100x8_vllm033dev450_cu130_ep/fixes.md).

| id | symptom | cause | fix |
| --- | --- | --- | --- |
| B2 | dense Llama: `Number of experts in the model must be greater than 0 when expert parallelism is enabled` | this build validates EP against the expert count | EP is N/A for dense models: `enable_expert_parallel=False` for Llama only; gate requires EP on for MoE |
| B3 | `FileNotFoundError .../triton/<hash>/_bincount_kernel.json` then `CompiledKernel has no attribute module` | JIT cache on encrypted FUSE `/root` | JIT caches on node-local disk |
| B4 | prefill final chunk "not found", memory snapshots empty | `assign_request_id` appends a random suffix to request ids | resolve the scheduler id right after `add_request` |
| B6 | memory per layer wrong inside V4's shared block-table group | per-layer page sizes differ within one group (37440/1728/8640 B); KV tensors overlay one buffer | use worker per-layer specs for page size; dedupe backing storages for slab bytes |
| B7 | MiMo FP8 KV: `fp8_kv_dequant requires fp16 or bf16 Q` at startup | FLASH_ATTN_DIFFKV upgrades to FA4 (diff-KV + sinks); its FP8-KV path needs a BF16/FP16 query | policy fallback to `auto` (BF16 KV); no backend forcing |
| B8 | GLM: bare `CUDA error: invalid argument` once, mid-grid (128K b2 prefill), worker died | unknown; did not reproduce in the unprobed control or the probed retry | GLM **does start and run with EP on** (TP8+EP, FP8 KV); retry with `--resume-sessions` carrying valid (point, rep) records |

Operational mistakes (do not repeat):
- **`pkill -f <pattern>` killed my own tool shell**, because the pattern matched the command line running it.
  Kill by exact PID from `ps -eo pid,cmd | awk '$2=="bash" && $3=="script.sh"{print $1}'`.
- **A background chain gated on "the smoke run wrote a report"** started a broken session. Gate on content
  (`smoke_gate.py`), not completion.
- **A wait pattern `finished {` also matched `child finished {`** and would have deleted a checkpoint before
  shutdown verification. Match `finished {'status'`.
- **The orchestrator didn't reap its child on the interrupt path.** The zombie leader kept `killpg(pgid, 0)`
  true, so shutdown wasn't verified for ~5 min. Fixed: `proc.poll()` before probing the group.
- **Reusing a session id** overwrote the earlier attempt's lifecycle/status/orchestrator logs. `session.py`
  now refuses an id that already has a lifecycle file.
- **Deleting a checkpoint deletes its `config.json`,** which the report needed. `checkpoint_inventory.py` now
  copies `config.json` into `raw/<model>/architecture/`, and `make_report.py` prefers that copy.
- **Assumed hardware/space numbers drift between containers** (`/tmp` free, GPU presence, node name). Re-check
  right before each download or launch.

---

## 8. Measured run times (launch → verified shutdown; excludes download, install and first-ever NFS load)

| model | primary session | matched overhead session | other |
| --- | ---: | ---: | --- |
| Qwen3.8 | 576 s (init 146, measure 349, shutdown 25) | 478 s | full-probe breakdown 566 s; subset overhead 344 s |
| Llama 3.1 70B | 435 s (init 65, measure 333, shutdown 7) | 410 s | default-mode 445 s; failed starts 240 s + 115 s |
| V4 0731 | 665 s (init 76, measure 550, shutdown 9) | 610 s | — |
| V4.1 | 565 s | 545 s | first engine start ~3 min (load 54 s from /tmp) |
| MiMo V2.6 | 495 s | 460 s | default-mode 495 s |

- One full model takes **≈ 15–22 min** of GPU time (primary + overhead).
- First-time costs come on top: Qwen weight load from NFS 229 s; V4 JIT warmup ~265 s; GLM's FlashInfer
  `fused_moe_90` build ~9 min.
- Per run: ~2 s at 1K, ~32 s at 128K batch 8 (mostly prefill).

---

## 9. Headline results (median of 3 independent reps, all 8 GPUs)

Decode attention per step (ms), batch 1 / batch 8, with attention share of the step at 64K batch 8:

| context | Llama | Qwen3.8 | V4 0731 | V4.1 (fp8 KV) | MiMo |
| ---: | --- | --- | --- | --- | --- |
| 1K | 3.71 / 4.00 | 1.70 / 2.03 | 3.50 / 3.53 | 3.48 / 3.67 | 1.93 / 2.34 |
| 16K | 4.27 / 5.84 | 1.78 / 2.17 | 3.61 / 3.96 | 3.51 / 3.76 | 2.01 / 2.61 |
| 64K | 5.53 / 11.38 | 1.86 / 2.33 | 3.75 / 4.22 | 3.57 / 3.93 | 2.25 / 3.40 |
| 128K | unsupported | 1.88 / 2.48 | 3.83 / 4.56 | 3.61 / 4.13 | 2.54 / 4.43 |
| share @64K b8 | 61 % | 28 % | 45 % | 37 % | 46 % |

Prefill attention for an 8K chunk ending at 16K / 64K / 128K (ms):
- Llama 146 / 370 / unsupported
- Qwen 62 / 70 / 80
- V4 116 / 165 / 239
- V4.1 63 / 72 / 84
- MiMo 40 / 70 / 110

Attention state at 128K, one request, settled decode (GiB, logical, all GPUs):
- Qwen 12.86 (incl. 4× KV-head replication)
- V4 3.90
- V4.1 1.78
- MiMo 5.65
- Llama 20.0 at 64K (128K unsupported)

The first three models are from the cu126 study, V4.1 and MiMo from cu130. Precision also differs by
model. Compare with care.

---

## 10. Open items
1. **GLM-5.3:** with EP on it starts and completes the full grid (study `2026-11-03_h100x8_vllm033dev450_cu130_ep`);
   one intermittent bare `CUDA error: invalid argument` remains unexplained (fixes.md B8). Bisect GLM custom kernels if it recurs.
2. **Decode attention without probe overhead** (needs cheaper markers or CUPTI attribution of graph kernels).
3. **Rerun Qwen, Llama and V4 in the cu130 env** if all six models must be one controlled set.
4. **Prefill GPU-busy time** (eager prefill windows include host launch gaps; matters most for the 1K chunk).
5. **Fix `install.sh` default `VLLM_VENV=$HOME/...`** (quota-full home) or always pass `VLLM_VENV`.

---

## 11. What to carry to the GitHub repo
- **Code:** `install.sh` (cu130 pin) and `bench/`: `env.sh`, `study_config.py`, `attn_probe.py`, `sched_probe.py`,
  `attn_study.py`, `analyze.py`, `session.py`, `smoke.py`, `smoke_gate.py`, `workload.py`, `download_model.py`,
  `checkpoint_inventory.py`, `make_report.py`. The `run_*.sh` chain scripts are session-specific examples.
- **Docs:** this file, plus `docs/gpu_setup_and_fixes.md`, `docs/setup_and_download_experience.md`,
  `docs/model_architectures.md` (Llama pin added).
- **Evidence:** `report/2026-11-01_h100x8_vllm033dev450/` (326 MB) and `report/2026-11-02_h100x8_vllm033dev450_cu130/` (148 MB).
  Both hold raw runs (gzip), CSVs, charts, `tables.md`, `fixes.md`, `model_completion.csv`, `manifest.json`.
  Decide whether raw runs go into git or an artifact store (AGENTS.md: large traces stay out of git).
- **Not in git:** weights (`/tmp/attn_study_weights/` is node-local; MiMo is still there, V4.1 was deleted),
  venvs, `/tmp` caches.

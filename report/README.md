# Study reports and evidence

All measurement outputs belong under `report/<study-id>/`. Create a fresh study ID for a new
runtime, node or protocol. Studies: [`2026-11-03_h100x8_vllm033dev450_cu130_ep`](2026-11-03_h100x8_vllm033dev450_cu130_ep/README.md)
(six models, TP8 + EP, FP8-first KV; 138 supported points × 3 valid repetitions, 6 Llama 128K points unsupported).

```text
report/<study-id>/
  README.md                 what completed, missing points, limits, reproduction commands
  manifest.json             frozen study identity, runtime/model pins, node and workload hashes
  config/                   installer copy, vllm-pin.txt, requirements.freeze.txt, launch configs
  logs/                     preflight, downloads, launches, retries, runner and shutdown logs
  raw/<model>/<experiment>/ repetition-1/, repetition-2/, repetition-3/, failed-attempts/
  architecture.csv          source-backed inventory (not a GPU timing experiment)
  architecture.md           six-model inventory table and source links
  memory.csv                all valid repetitions and explicit failed/unavailable statuses
  memory_layers.csv         per-rank, per-owner/layer state observations
  decode.csv                all decode repetitions
  prefill.csv               all prefill repetitions
  summary.csv               median, min, max, valid count and raw repetition references
  model_completion.csv      per-model start/completion timestamps, duration and final status
  charts/                   memory.png, decode.png, prefill.png generated from saved CSVs
  fixes.md                  bug symptoms, cause, fix and verification, if any bugs occurred
  artifacts.json            external large-artifact locations and SHA-256 hashes, if needed
```

Use **three independent valid repetitions for every supported GPU measurement point** in memory,
decode and prefill. Independently rebuild requests/state for each repetition; three steps from one
capture are not three repetitions. Retain per-step samples inside each raw run. The summary reports
the median plus min/max and `valid_repeats=3`, linked to all three raw runs. Do not select the fastest
run or imply three repetitions guarantee accuracy. Warmup, matched setup and instrumentation checks
still apply. Preserve failed attempts; they never count as valid repetitions. If only one or two valid
runs exist, retain them and mark the point incomplete, with the reason.

## Per-model completion time

Write one row per model session/attempt in `model_completion.csv`:

```text
study_id,model,revision,session_id,started_at_utc,measurement_completed_at_utc,
shutdown_verified_at_utc,finished_at_utc,elapsed_seconds,valid_points,expected_points,
valid_repetitions,expected_repetitions,status,reason,manifest,log
```

Use ISO 8601 UTC timestamps with `Z`. `started_at_utc` is immediately before that model's launch;
`measurement_completed_at_utc` is when its last required measurement/inventory check ends;
`shutdown_verified_at_utc` is when owned workers have exited and GPU usage has returned to baseline.
`finished_at_utc` is the session's terminal timestamp, even if failed/interrupted. Measure elapsed
seconds using a monotonic clock from launch start through terminal cleanup. This duration excludes
checkpoint download and environment installation, which have their own logs/timestamps.

Only mark a model `complete` when all supported points have three valid repetitions, every excluded
point has a documented reason, evidence is saved, and shutdown is verified. Missing/unsupported
measurements remain visible even in a finished session. Use `partial`, `failed`, or `interrupted`
otherwise; leave unavailable completion/shutdown timestamps empty. Append a new session row for a
retry/resume; never overwrite an earlier attempt or invent a completed timestamp.
Show per-model completion timestamps, duration, status and missing points in the study README.

## Evidence and fixes

Keep original raw bytes and failed logs. Generate CSV summaries and charts using saved evidence;
record commands and code version. Config includes TP/EP, dtypes, attention backend, precision,
chunk/model limits, actual input/batch lengths and instrumentation. Record every repeat separately.

If a bug occurs, `fixes.md` records the symptom, affected model/run IDs, linked logs, cause (or unknown),
exact change/patch, runtime/config versions, verification and whether affected measurements need
rerunning. A corrected run gets a new attempt ID; the old evidence stays intact. Promote a reusable
lesson into `docs/` and link back to this study's fix rather than replacing the original incident.

Keep credentials/private addresses out of committed evidence. Large traces remain out of Git:
store them under the study's `raw/traces/` on the GPU node or an authorized artifact store and index
them in `artifacts.json`. All result records and references still live within this study directory.
Do not commit or push without request.

## Current launch policy evidence

Every model's `config/` and `logs/` must retain the first explicit FP8-KV attempt, requested/resolved
KV dtype/format, effective EP setting, weight/recurrent/index/tail dtypes, scale source and warnings.
If FP8 fails, save its error and the reason for fallback to `auto` or another supported runnable
choice. Validate the final configuration before accepted repetitions; never aggregate repeats from
different settings. Use TP8 + EP on (dense Llama N/A) and the frozen cu130 environment for this new
comparison. Keep historical cu126/cu130/BF16-Qwen study folders unchanged.

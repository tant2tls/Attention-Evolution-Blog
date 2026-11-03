# Attention evolution: GPU measurement workspace

Read [AGENTS.md](AGENTS.md), then [attention_measure_plan.md](attention_measure_plan.md).
The current deliverable is real GPU evidence for architecture inventory, attention-state memory,
decode attention time and prefill attention time. The existing `index.html` is a synthetic prototype;
its numbers are not evidence. Write the final blog after measurements are reviewed.

| Location | Purpose |
| --- | --- |
| `install.sh` | Linux GPU installer: exact latest-observed vLLM version/commit and environment capture |
| `attention_measure_plan.md` | Single experiment plan and acceptance rules |
| `docs/model_architectures.md` | Four inherited checkpoint pins plus official Qwen FP8; Llama remains to pin |
| `docs/gpu_setup_and_fixes.md` | Relevant setup, previous failures and fixes, plus instrumentation gaps |
| `docs/import_provenance.json` | Source paths and SHA-256 hashes for this selective transfer |
| `bench/trace_breakdown.py`, `bench/blog_layers.py` | Historical offline trace inspection aids, not measurement runners |
| `bench/attn_study.py`, `attn_probe.py`, `session.py`, … | Study runners (see [bench/README.md](bench/README.md)) |
| `bench/README.md` | Limits of those aids and requirements for the new runners |
| `report/<study-id>/` | All study results, raw repetitions, configs, logs, charts, completion times and bug fixes |
| [report/2026-11-03_h100x8_vllm033dev450_cu130_ep](report/2026-11-03_h100x8_vllm033dev450_cu130_ep/README.md) | Measured study: six models, memory/decode/prefill, fixes B1–B8 |
| [report/README.md](report/README.md) | Study folder layout, three-repeat rule and completion records |
| [docs/README.md](docs/README.md) | Reusable setup, checkpoint download and workflow experience |

No Frontier results, publishing automation, broad serving sweeps or speculative experiments were imported.
Historical sessions and open failures are recorded in [docs/experience.md](docs/experience.md).
They mix CUDA environments and use BF16 Qwen; they are not the newly configured controlled study.
Read that experience before implementing or launching the new sessions.

## Install the frozen latest-observed runtime

On the authorized Linux GPU node, run `bash install.sh`. The installer pins vLLM
`0.33.0rc1.dev450+g17e675965` at commit `17e675965d54bef8a3eb77ea92c6bae195ef793c`,
the latest nightly listed by the official wheel index when checked on 2026-11-02.
Set `VLLM_VENV` to a fresh scratch/local path (default: `./.venv-attention`). The script activates it,
installs the pinned wheel, checks the version/dependencies, and saves `requirements.freeze.txt`
and `vllm-pin.txt` inside that environment. After installation, run
`source .venv-attention/bin/activate` in your shell (adjust if you overrode the path).
Copy those two records into `report/<study-id>/config/` before measuring; collect GPU/driver
and CUDA details during node preflight. Use a new environment for any new runtime study.
Six-model compatibility and complete launch configuration still need validation before measurements.
For reproducing resolved dependencies, use the saved freeze and the same commit index on matching
hardware/Python; the installer pins vLLM and torch 2.14.0+cu130, while the full dependency
resolution is recorded on the node. Check driver/forward compatibility before GPU measurement.

Every supported memory, decode and prefill point requires **three independent valid repetitions**.
Save all raw runs, report the median and min/max, and preserve failed attempts separately.
Each model session records its start, measurement completion, verified shutdown, elapsed time and
status in `report/<study-id>/model_completion.csv`. See [report rules](report/README.md).
Reusable lessons belong in `docs/`; study-specific fixes and their evidence stay in the report.

Qwen uses the official `Qwen/Qwen3.8-Flash-Next-FP8`, revision `236dfdf285828023ca3bcd3f37366c58a3469b13`,
replacing the inherited BF16 choice. See the architecture document for mixed-precision and state-dtype
caveats; this is still a six-model deployment comparison.

Blog scope (2026-11-03): the measured study already supplies every blog number; remaining work is offline
(see [plan §0](attention_measure_plan.md)). Rerun GPU work only for a listed trigger, with `ATTN_DECODE_BATCHES=1,8`.

Execution policy (2026-11-03): **TP8 + EP on**, utilization **0.82**, Qwen's official FP8 checkpoint.
For each model validate **`--kv-cache-dtype fp8` first**; if it fails, preserve the failure, try `auto`,
then another supported runnable choice. Record actual state formats, scale warnings and every
fallback, and keep one selected configuration per model's three valid repetitions. See the plan
for smoke/overhead checks and the recorded V4.1 EP-off failure that motivated this policy.

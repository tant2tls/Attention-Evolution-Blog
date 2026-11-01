# Attention evolution: GPU measurement workspace

Read [AGENTS.md](AGENTS.md), then [attention_measure_plan.md](attention_measure_plan.md).
The current deliverable is real GPU evidence for architecture inventory, attention-state memory,
decode attention time and prefill attention time. The existing `index.html` is a synthetic prototype;
its numbers are not evidence. Write the final blog after measurements are reviewed.

| Location | Purpose |
| --- | --- |
| `install.sh` | Linux GPU installer: exact latest-observed vLLM version/commit and environment capture |
| `attention_measure_plan.md` | Single experiment plan and acceptance rules |
| `docs/model_architectures.md` | Five inherited checkpoint identities and attention implementation leads; Llama remains to pin |
| `docs/gpu_setup_and_fixes.md` | Relevant setup, previous failures and fixes, plus instrumentation gaps |
| `docs/import_provenance.json` | Source paths and SHA-256 hashes for this selective transfer |
| `bench/trace_breakdown.py`, `bench/blog_layers.py` | Historical offline trace inspection aids, not measurement runners |
| `bench/README.md` | Limits of those aids and requirements for the new runners |
| `report/<study-id>/` | All study results, raw repetitions, configs, logs, charts, completion times and bug fixes |
| [report/README.md](report/README.md) | Study folder layout, three-repeat rule and completion records |
| [docs/README.md](docs/README.md) | Reusable setup, checkpoint download and workflow experience |

No Frontier results, publishing automation, broad serving sweeps or speculative experiments were imported.
No GPU session has been run for this study. See the setup document before implementing the runners.

## Install the frozen latest-observed runtime

On the authorized Linux GPU node, run `bash install.sh`. The installer pins vLLM
`0.33.0rc1.dev450+g17e675965` at commit `17e675965d54bef8a3eb77ea92c6bae195ef793c`,
the latest nightly listed by the official wheel index when checked on 2026-11-02.
It creates `~/vllm-attention` (override with `VLLM_VENV`), activates it inside the script,
installs the pinned wheel, checks the version/dependencies, and saves `requirements.freeze.txt`
and `vllm-pin.txt` inside that environment. After installation, run
`source ~/vllm-attention/bin/activate` in your shell (adjust if you overrode the path).
Copy those two records into `report/<study-id>/config/` before measuring; collect GPU/driver
and CUDA details during node preflight. Use a new environment for any new runtime study.
Six-model compatibility and complete launch configuration still need validation before measurements.
For reproducing resolved dependencies, use the saved freeze and the same commit index on matching
hardware/Python; the installer alone pins vLLM, while dependency resolution is recorded on the node.

Every supported memory, decode and prefill point requires **three independent valid repetitions**.
Save all raw runs, report the median and min/max, and preserve failed attempts separately.
Each model session records its start, measurement completion, verified shutdown, elapsed time and
status in `report/<study-id>/model_completion.csv`. See [report rules](report/README.md).
Reusable lessons belong in `docs/`; study-specific fixes and their evidence stay in the report.

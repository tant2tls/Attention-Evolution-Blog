# Attention measurement tools

Only two dependency-free, offline inspection aids were copied byte-for-byte from Frontier:

- `trace_breakdown.py TRACE.json.gz --csv OUT.csv`: historical kernel categories and per-step sums.
  It uses GPU annotation spans and a legacy classifier. It may miss multi-stream/graph work,
  misclassify new kernels and select one rank/thread. Never use its sums as attention elapsed time.
- `blog_layers.py TRACE.json.gz --pattern REGEX --json`: compares long and short matching calls
  in the last full prefill step. The default matches expert GEMMs as a diagnostic proxy for trimmed
  layers. This is a heuristic, not measured attention timing or exact executed-position accounting.

Both have `--help` and require only Python's standard library. They read existing traces; they do
not launch a server or generate measurements. Full raw traces stay outside Git.

Implement new study-specific memory/decode/prefill runners only after reading the plan and setup
lessons. They need six-model manifests, safe server ownership, exact state-owner introspection,
cross-stream GPU timing, actual workload validation and failure preservation. No old broad serving
runner, prefix/speculation arm, auto-commit publisher or installer was copied.

All new runners write to `report/<study-id>/` as defined in [report rules](../report/README.md).
Keep config, logs, per-model raw repetitions, CSVs, summaries, charts, completion times and any
bug-fix guide together there. Every supported point requires three independent valid repetitions;
summary medians/min/max link to all raw runs. Record model completion/cleanup timestamps even
when a session fails or is interrupted. General setup/download lessons belong in `docs/`.

## Study runners (2026-11-03)

| File | Role |
| --- | --- |
| `study_config.py` | Frozen study id, model pins, common TP8/EP/0.82 policy, per-model engine/env deviations |
| `env.sh` | Activate the frozen venv; study-owned `HF_HOME`; JIT caches on node-local disk |
| `download_model.py` | Pinned-revision download into a study-owned dir with per-file size check and log |
| `checkpoint_inventory.py` | Safetensors-header parameter inventory and config-based attention-layer counts |
| `workload.py` | Public Gutenberg corpus → per-model raw token-ID prompts with hashes |
| `attn_probe.py` | vLLM worker extension: replay-valid CUDA events per attention module and model window, state-buffer introspection |
| `attn_study.py` | One engine launch: smoke / study / overhead modes, step-driven decode, prefill and co-collected memory, `--resume-sessions` |
| `session.py` | Process-group ownership, timeout, cleanup, GPU-idle shutdown verification, `model_completion.csv` row |
| `freeze_config.py` | Write-once per-model selected deployment (`config/<model>__selected.json`) |
| `run_accepted.sh` | Accepted session followed by matched unprobed control (one model) |
| `run_queue.sh` | Recommended: two-phase unattended queue (validate + freeze all six, then accepted + control in reverse order, one resumed retry); see plan §7a |
| `analyze.py`, `make_report.py`, `write_readme.py` | CSVs/summary/hashes, charts (separate matplotlib venv), study README + manifest |

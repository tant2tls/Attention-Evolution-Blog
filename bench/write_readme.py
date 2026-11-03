"""Regenerate report/<study>/README.md from saved CSVs, selected configs and completion records."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402

CTX = {"1024": "1K", "16384": "16K", "65536": "64K", "131072": "128K"}


def rd(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def f(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return x or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    st = Path(a.study_dir)
    summ = rd(st / "summary.csv")
    idx = {(r["experiment"], r["model"], r["context_tokens"], r.get("target_batch", ""), r["metric"]): r for r in summ}
    comp = rd(st / "model_completion.csv")
    L = []
    L.append(f"# Study `{st.name}`\n")
    L.append("Attention-state memory, decode attention time and prefill attention time for six checkpoints on one "
             "8 × H100 80GB node, vLLM `0.33.0rc1.dev450+g17e675965` (commit `17e675965d54…`), torch "
             "`2.14.0+cu130`, driver 580.173.02. TP8, EP on for MoE models (N/A for dense Llama), "
             "`gpu_memory_utilization=0.82`, prefix caching and speculation/MTP off, 8192-token prefill budget, "
             "`max_num_seqs=8`, synchronous scheduling, FP8 KV first. Raw token-ID prompts cut from public-domain "
             "Gutenberg texts (`bench/workload.py`). Medians of three independent repetitions; "
             "min/max and raw run paths are in `summary.csv`. This is a deployment comparison: KV/state dtypes, "
             "backends and graph modes differ by model and are listed below.\n")
    if a.notes:
        L.append(a.notes + "\n")
    L.append("## Model status\n")
    L.append("| model | config id | KV requested → resolved | EP | accepted / control sessions | status |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for k in sc.MODEL_ORDER:
        p = st / "config" / f"{k}__selected.json"
        if p.exists():
            s = json.loads(p.read_text())
            L.append(f"| {sc.MODELS[k]['display']} | `{s['config_id']}` | {s['kv_dtype_requested']} → "
                     f"{s['kv_format']} | {s['ep_effective']} | {', '.join(s['accepted_sessions']) or '-'} / "
                     f"{', '.join(s['overhead_sessions']) or '-'} | see completion table |")
        else:
            L.append(f"| {sc.MODELS[k]['display']} | not frozen | | | | pending / not measured |")
    L.append("\n## Per-model session completion (`model_completion.csv`)\n")
    L.append("| model | session | started (UTC) | measurement completed | shutdown verified | elapsed s | "
             "valid points / expected | status | reason |")
    L.append("| --- | --- | --- | --- | --- | ---: | --- | --- | --- |")
    for r in comp:
        L.append(f"| {r['model']} | {r['session_id']} | {r['started_at_utc']} | {r['measurement_completed_at_utc']} | "
                 f"{r['shutdown_verified_at_utc']} | {r['elapsed_seconds']} | {r['valid_points']} / "
                 f"{r['expected_points']} | {r['status']} | {r['reason']} |")
    L.append("\nValidation (smoke) sessions have expected points 0; their status reflects only the validation gate.\n")
    L.append("## Decode attention, ms per step (median of 3; batch 1 / batch 8)\n")
    L.append("| context | " + " | ".join(sc.MODELS[k]["display"] for k in sc.MODEL_ORDER) + " |")
    L.append("| --- |" + " ---: |" * len(sc.MODEL_ORDER))
    for c in ["1024", "16384", "65536", "131072"]:
        cells = []
        for k in sc.MODEL_ORDER:
            v = []
            for b in ("1", "8"):
                r = idx.get(("decode", k, c, b, "attention_ms_per_step"))
                v.append(f(r["median"]) if r and r["median"] else (r["status"] if r else "–"))
            cells.append(" / ".join(v))
        L.append(f"| {CTX[c]} | " + " | ".join(cells) + " |")
    L.append("\n## Prefill attention, ms per chunk (median of 3)\n")
    L.append("| chunk | " + " | ".join(sc.MODELS[k]["display"] for k in sc.MODEL_ORDER) + " |")
    L.append("| --- |" + " ---: |" * len(sc.MODEL_ORDER))
    for c in ["1024", "16384", "65536", "131072"]:
        lab = "1K first chunk (1024)" if c == "1024" else f"8K ending at {CTX[c]}"
        cells = []
        for k in sc.MODEL_ORDER:
            r = idx.get(("prefill", k, c, "", "attention_ms_per_chunk"))
            cells.append(f(r["median"], 1) if r and r["median"] else (r["status"] if r else "–"))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("\n## Attention-state memory, GiB for one request, all 8 GPUs (settled decode, occupied blocks)\n")
    L.append("| context | " + " | ".join(sc.MODELS[k]["display"] for k in sc.MODEL_ORDER) + " |")
    L.append("| --- |" + " ---: |" * len(sc.MODEL_ORDER))
    for c in ["1024", "16384", "65536", "131072"]:
        cells = []
        for k in sc.MODEL_ORDER:
            r = idx.get(("memory", k, c, "", "total_attention_state_gib"))
            cells.append(f(r["median"], 3) if r and r["median"] else (r["status"] if r else "–"))
        L.append(f"| {CTX[c]} | " + " | ".join(cells) + " |")
    ov = rd(st / "overhead.csv")
    if ov:
        L.append("\n## Probe overhead (matched unprobed control, whole-model ms)\n")
        L.append("| model | experiment | context | batch | probed | unprobed | overhead % | attention lower bound ms |")
        L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in ov:
            L.append(f"| {r['model']} | {r['experiment']} | {CTX.get(r['context_tokens'], r['context_tokens'])} | "
                     f"{r['target_batch']} | {f(r['probed_model_ms'])} | {f(r['unprobed_model_ms'])} | "
                     f"{f(r['overhead_pct'], 1)} | {f(r.get('attention_lower_bound_ms', ''))} |")
    L.append("\n## Definitions and limits\n")
    L.append("""- **Attention time**: CUDA events (`external=True`, re-recorded on every CUDA-graph replay) at entry and
  exit of each decoder layer's attention module (projections, core, sparse index/compression, recurrent
  update, cache writes; FFN/MoE excluded). Per rank the per-layer times are summed; the node value is the
  maximum over the 8 ranks. Steps where any event was stale or outside that step's model window are rejected.
- **Whole-model time**: model-forward window + logits window on each rank; node value = max over ranks.
  Attention share uses these two compatible event-based quantities from the same (probed) steps.
- **Probe overhead**: module-level events add host/graph work. The matched unprobed session (same frozen
  config, model/logits windows only) gives the overhead; `attention_lower_bound_ms` = attention − overhead.
- **Decode**: B distinct prompts of exactly L tokens; after prefill, 50 untimed pure-decode steps with exactly
  B sequences, then 50 measured consecutive pure-decode steps. Requests that finish prefill early keep decoding
  while the others prefill, so per-request context differs slightly; first/last contexts are in `decode.csv`.
- **Prefill**: the step whose 8192-token chunk ends at L (1024-token first chunk at 1K), measured eagerly.
  `executed_positions_by_layer` records the token count each attention module received.
- **Memory**: live blocks of the single request in every KV-cache group (scheduler block tables, which see
  sliding-window frees) × each layer's runtime page size, summed over layers and the 8 ranks. Shared/KV-sharing
  consumer layers own nothing (listed as consumers in `memory_layers.csv`). `logical_payload_gib` is the
  token-proportional payload inside those pages; `slab_gib` is what the shared block pool reserves for the
  request's block ids; `gpu_allocation_delta_gib` is a sanity check only (pools are preallocated).
- Labels: every CSV number is `measured` unless marked; architecture numbers are `calculated from checkpoint`.
""")
    (st / "README.md").write_text("\n".join(L))
    print("README written")


if __name__ == "__main__":
    main()

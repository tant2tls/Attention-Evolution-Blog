"""Build study CSVs and summary from saved raw evidence only (no GPU, stdlib only).

Inputs : report/<study>/raw/<model>/{memory,decode,prefill}/{repetition-*,failed-attempts,unsupported}/*.json
         report/<study>/raw/<model>/sessions/<session>/{resolved_config,probe_info,scheduler_kv_cache_config}.json
         report/<study>/config/<model>__selected.json (frozen per-model deployment)
Outputs: memory.csv, memory_layers.csv, decode.csv, prefill.csv, overhead.csv, summary.csv,
         architecture.csv, architecture.md, manifest_hashes.json
Only sessions listed as accepted/overhead in <model>__selected.json enter the CSVs; everything
else (smoke, failed attempts) remains in raw/ and logs/ for audit.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402

GIB = 1024 ** 3
MIB = 1024 ** 2
EXTRA = ["config_id", "ep_requested", "ep_effective", "kv_dtype_requested", "kv_dtype_resolved", "kv_format",
         "state_dtype_evidence", "kv_scale_source", "fallback_reason"]


def load(p: Path):
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as f:
            return json.loads(f.read())
    return json.loads(p.read_text())


def selected(study: Path, key: str):
    p = study / "config" / f"{key}__selected.json"
    return load(p) if p.exists() else None


def extra_cols(sel: dict) -> dict:
    return {k: sel.get(k, "") for k in EXTRA}


def points(study: Path, key: str, exp: str, sessions: list[str]):
    base = study / "raw" / key / exp
    out = []
    for sub in ("repetition-1", "repetition-2", "repetition-3", "failed-attempts", "unsupported"):
        for f in sorted(glob.glob(str(base / sub / "*.json")) + glob.glob(str(base / sub / "*.json.gz"))):
            name = Path(f).name
            sess = name.split("__")[0]
            if sess not in sessions:
                continue
            d = load(Path(f))
            d["_path"] = str(Path(f).relative_to(study))
            out.append(d)
    return out


# ----------------------------------------------------------------------------- memory
def classify(spec: dict, layer: str) -> str:
    """Map a runtime KV spec + layer name onto the plan's disjoint state categories."""
    spec = spec or {}
    cls = spec.get("__cls__", "")
    n = layer.lower()
    if n.endswith(".ple"):  # Qwen per-layer-embedding buffer registered as a Mamba-type cache: not attention
        return "non_attention_state"
    if "Mamba" in cls:
        return "recurrent"
    if "indexer" in n or ".index" in n or "Indexer" in cls:
        return "sparse_index_tail"
    if "compressor" in n or "state_cache" in n or "tail" in n:
        return "sparse_index_tail"
    if "SlidingWindow" in cls or "CircularBuffer" in cls or "RSWA" in cls or spec.get("sliding_window"):
        return "window_kv"
    try:
        tps = float(spec.get("tokens_per_state") or 1)
    except (TypeError, ValueError):
        tps = 1.0
    if tps > 1:
        return "shared_compressed"
    return "ordinary_kv"


def memory_rows(study: Path, key: str, sel: dict):
    """One row per memory snapshot plus per-layer owner/consumer rows.

    Per rank, a layer owns n_live(group) whole pages of page_size_bytes (scheduler block table of the
    single request x the runtime spec). Every rank holds an identically shaped shard, so node bytes =
    per-rank bytes x TP (replicated KV heads are physical copies and are counted). KV-sharing consumer
    layers own nothing and reference their target. Slab bytes = |union of live block ids| x
    (sum of KV tensor sizes / num_blocks) x TP: what the shared block pool actually reserves.
    """
    if not sel.get("accepted_sessions"):
        return [], []
    sess = sel["accepted_sessions"][0]
    sdir = study / "raw" / key / "sessions" / sess
    kvc = load(sdir / "scheduler_kv_cache_config.json")
    tp = load(sdir / "resolved_config.json")["tp"]
    try:
        bufs = load(sdir / "state_buffers.json.gz")
        r0 = [b for b in bufs if b.get("rank") == 0][0]["layers"]
    except Exception:
        r0 = {}
    consumers = {n: v["kv_sharing_target_layer_name"] for n, v in r0.items()
                 if v.get("kv_sharing_target_layer_name")}
    # Model-internal sharing (V4.1 kv_source_layer_ids): attention modules with no cache bytes of their own
    # read state owned by source layers. Exact source per consumer is inferred from config, not traced.
    try:
        acfg = json.loads((study / "raw" / key / "architecture" / "config.json").read_text())
        acfg = acfg.get("text_config", acfg)
    except Exception:
        acfg = {}
    srcs = acfg.get("kv_source_layer_ids")
    if srcs:
        import re as _re
        ratios = acfg.get("compress_ratios", [])
        for n, v in r0.items():
            m = _re.search(r"layers\.(\d+)\.attn$", n)
            if not m:
                continue
            i = int(m.group(1))
            own = sum(t.get("nbytes", 0) for t in v.get("tensors", []))
            if own == 0 and i not in srcs and i < acfg.get("num_hidden_layers", 0) and \
                    (i < len(ratios) and ratios[i] != 0):
                prev = [x for x in srcs if x <= i]
                consumers[n] = (f"layers.{prev[-1]}.attn (inferred from kv_source_layer_ids {srcs}; not traced)"
                                if prev else f"kv_source_layer_ids {srcs} (inferred)")
    category_override = dict(sel.get("state_category_by_layer", {}))
    if srcs:  # source layers own state read by other layers: "compressed / shared KV" even if uncompressed
        for n in r0:
            for i in srcs:
                if n.endswith(f"layers.{i}.attn"):
                    category_override[n] = "shared_compressed"
    groups = kvc.get("kv_cache_groups", [])
    # Pool bytes reserved per block id: unique backing storages of all layer cache tensors (KVCacheTensor
    # entries can be overlapping views of one block-outer buffer, so their sizes must not be summed).
    storages = {}
    for v in r0.values():
        for t in v.get("tensors", []):
            if t.get("storage_nbytes"):
                storages[t["storage_ptr"]] = t["storage_nbytes"]
    pool_bytes = sum(storages.values()) or sum(t["size"] for t in kvc.get("kv_cache_tensors", []))
    per_block_bytes = pool_bytes / max(kvc.get("num_blocks", 1), 1)
    rows, lrows = [], []
    for d in points(study, key, "memory", sel["accepted_sessions"]):
        base = dict(model=key, revision=sc.MODELS[key]["revision"], context_tokens=d.get("context_tokens"),
                    repeat=d.get("repeat"), status=d.get("status"), manifest=d["_path"], **extra_cols(sel))
        if d.get("status") != "measured":
            rows.append(dict(base, snapshot_phase="", total_attention_state_gib=d.get("status"),
                             why="; ".join(d.get("why", []))[:300]))
            continue
        for snap in d["snapshots"]:
            rq = list(snap["reqs"].values())[0]
            live_tokens = rq["num_computed_tokens"]
            cats = dict(ordinary_kv=0, window_kv=0, shared_compressed=0, sparse_index_tail=0, recurrent=0)
            other = 0
            logical = 0.0
            union = set()
            for gi, g in enumerate(groups):
                spec = g.get("kv_cache_spec") or {}
                gl = rq["groups"][gi]
                nlive = gl["n_live"]
                union.update(gl.get("live_ids", []))
                specs = spec.get("kv_cache_specs") or {}
                for ln in g.get("layer_names", []):
                    # Per-layer runtime spec (worker get_kv_cache_spec) wins: page sizes differ by layer
                    # inside one block-table group (fixes.md B6). Fall back to the group spec.
                    lsp = (r0.get(ln) or {}).get("spec")
                    ls = lsp if isinstance(lsp, dict) and isinstance(lsp.get("page_size_bytes"), int) \
                        else specs.get(ln, spec)
                    page = ls.get("page_size_bytes") if isinstance(ls.get("page_size_bytes"), int) else 0
                    real = ls.get("real_page_size_bytes") if isinstance(ls.get("real_page_size_bytes"), int) else page
                    cat = category_override.get(ln) or classify(ls, ln)
                    owned = nlive * page
                    bs = ls.get("block_size") or 1
                    tps = ls.get("tokens_per_state") or 1
                    try:
                        tps = float(tps)
                    except (TypeError, ValueError):
                        tps = 1.0
                    if cat == "recurrent":
                        lb = real if nlive else 0
                    else:
                        tps = tps if tps > 0 else 1.0
                        per_block = max(bs / tps, 1.0)  # stored entries per block (block_size counts tokens)
                        toks = live_tokens
                        win = ls.get("sliding_window")
                        if isinstance(win, int) and win > 0:
                            toks = min(toks, win)
                        lb = min(toks / tps, nlive * per_block) * real / per_block
                    if cat == "non_attention_state":
                        other += owned * tp
                    else:
                        cats[cat] += owned * tp
                        logical += lb * tp
                    lrows.append(dict(model=key, revision=sc.MODELS[key]["revision"],
                                      context_tokens=d["context_tokens"], layer_id=ln, attention_type=ls.get("__cls__"),
                                      state_owner_id=f"group{gi}:{ln}", owner_or_consumer="owner",
                                      unique_owned_mib=round(owned / MIB, 6), shared_reference="",
                                      rank="per rank (x%d ranks)" % tp, state_category=cat, live_blocks=nlive,
                                      page_bytes=page, repeat=d["repeat"], status="measured",
                                      snapshot_phase=snap["phase"], manifest=d["_path"], **extra_cols(sel)))
            for ln, tgt in sorted(consumers.items()):
                lrows.append(dict(model=key, revision=sc.MODELS[key]["revision"], context_tokens=d["context_tokens"],
                                  layer_id=ln, attention_type=r0.get(ln, {}).get("cls", ""), state_owner_id="",
                                  owner_or_consumer="consumer", unique_owned_mib=0, shared_reference=tgt,
                                  rank="per rank", state_category="shared reference (counted at owner)",
                                  repeat=d["repeat"], status="measured", snapshot_phase=snap["phase"],
                                  manifest=d["_path"], **extra_cols(sel)))
            occupied = sum(cats.values())
            mb = d.get("mem_baseline") or []
            mm = snap.get("mem") or []
            try:
                delta = sum(b["total"] - b["free"] for b in mm) - sum(b["total"] - b["free"] for b in mb)
            except Exception:
                delta = None
            rows.append(dict(base, snapshot_phase=snap["phase"], actual_live_tokens=live_tokens,
                             ordinary_kv_gib=cats["ordinary_kv"] / GIB, window_kv_gib=cats["window_kv"] / GIB,
                             unique_kv_gib=(cats["ordinary_kv"] + cats["window_kv"]) / GIB,
                             shared_compressed_gib=cats["shared_compressed"] / GIB,
                             sparse_index_tail_gib=cats["sparse_index_tail"] / GIB,
                             recurrent_state_gib=cats["recurrent"] / GIB,
                             total_attention_state_gib=occupied / GIB,
                             occupied_block_gib=occupied / GIB,
                             logical_payload_gib=logical / GIB,
                             slab_gib=len(union) * per_block_bytes * tp / GIB,
                             non_attention_state_gib=other / GIB,
                             gpu_allocation_delta_gib=(delta / GIB) if delta is not None else "unavailable",
                             reconciles=abs(sum(cats.values()) - occupied) < 1, why=""))
    return rows, lrows


# ----------------------------------------------------------------------------- decode/prefill
def decode_rows(study, key, sel, sessions, probe=True):
    rows = []
    for d in points(study, key, "decode", sessions):
        rows.append(dict(model=key, revision=sc.MODELS[key]["revision"], context_tokens=d.get("context_tokens"),
                         target_batch=d.get("target_batch") or int(d["point"].split("_B")[1]),
                         actual_batch=(d.get("actual_batch") or [""])[0] if d.get("actual_batch") else "",
                         attention_ms_per_step=d.get("attention_ms_per_step", "unavailable" if probe else ""),
                         model_ms_per_step=d.get("model_ms_per_step", ""),
                         effective_attention_tokens_per_s=d.get("effective_attention_tokens_per_s", ""),
                         context_start_tokens=d.get("context_start_tokens", ""),
                         context_end_tokens=d.get("context_end_tokens", ""),
                         warmup_steps=d.get("warmup_steps", ""), measured_steps=d.get("measured_steps", ""),
                         timing_method="CUDA events (external, graph-replay valid) around each layer's attention "
                                       "module; per-rank sum over layers, node=max over 8 ranks; whole model="
                                       "forward+logits windows, node=max over ranks",
                         repeat=d.get("repeat"), status=d.get("status"), why="; ".join(d.get("why", []))[:300],
                         session=d.get("session"), manifest=d["_path"], **extra_cols(sel)))
    return rows


def prefill_rows(study, key, sel, sessions, probe=True):
    rows = []
    for d in points(study, key, "prefill", sessions):
        hist = d.get("executed_positions_hist")
        rows.append(dict(model=key, revision=sc.MODELS[key]["revision"], context_tokens=d.get("context_tokens"),
                         new_chunk_tokens=d.get("new_chunk_tokens"),
                         executed_positions_by_layer=json.dumps(hist) if hist else ("unavailable" if probe else ""),
                         attention_ms_per_chunk=d.get("attention_ms_per_chunk", "unavailable" if probe else ""),
                         model_ms_per_chunk=d.get("model_ms_per_chunk", ""),
                         timing_method="CUDA events around attention modules in the final-chunk step (eager); "
                                       "node=max over ranks of per-rank layer sums",
                         repeat=d.get("repeat"), status=d.get("status"), why="; ".join(d.get("why", []))[:300],
                         session=d.get("session"), manifest=d["_path"], **extra_cols(sel)))
    return rows


def summarize(rows, keys, metrics, exp):
    groups = {}
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        ok = [r for r in rs if r.get("status") == "measured"]
        st = {r.get("status") for r in rs}
        for met in metrics:
            vals = [float(r[met]) for r in ok if r.get(met) not in ("", None, "unavailable")]
            base = dict(zip(keys, k))
            status = ("complete" if len(vals) >= 3 else
                      "unsupported" if st == {"unsupported"} else
                      "incomplete" if vals else
                      ("unavailable" if ok else "failed"))
            out.append(dict(base, experiment=exp, metric=met,
                            median=statistics.median(vals) if vals else "",
                            min=min(vals) if vals else "", max=max(vals) if vals else "",
                            valid_repeats=len(vals), status=status,
                            label="measured" if vals else status,
                            raw_ids=" ".join(r["manifest"] for r in ok),
                            config_id=rs[0].get("config_id", "")))
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def architecture(study: Path, keys):
    rows = []
    for key in keys:
        p = study / "raw" / key / "architecture" / "inventory.json"
        if not p.exists():
            rows.append(dict(model=key, status="unverified"))
            continue
        inv = json.loads(p.read_text())
        sel = selected(study, key) or {}
        rows.append(dict(model=key, revision=inv["revision"],
                         total_params_b=round(inv["total_params"] / 1e9, 2),
                         base_model_params_b=round(inv["base_model_params"] / 1e9, 2),
                         active_decode_params_b=round(inv["active_params_per_token"] / 1e9, 2),
                         active_prefill_params_b=round(inv["active_params_per_token"] / 1e9, 2),
                         lookup_table_params_b=round(inv["categories"].get("lookup_table", {}).get("params", 0) / 1e9, 2),
                         mtp_draft_params_b=round((inv["categories"].get("mtp", {}).get("params", 0) +
                                                   inv["categories"].get("draft", {}).get("params", 0)) / 1e9, 2),
                         layers=inv["layers"],
                         attention_types_and_counts="; ".join(f"{k}: {v}" for k, v in inv["attention"]["types"].items()),
                         state_types=sel.get("state_types", "see memory_layers.csv"),
                         dtype="; ".join(f"{k}: {v['params'] / 1e9:.1f}B" for k, v in inv["dtypes"].items()),
                         kv_state_dtype=(sel.get("kv_format", "").split("x")[0] if sel else "")
                         .replace("auto", "auto = BF16" if sel and "bfloat16" in sel.get("state_dtype_evidence", "")
                                  and sel.get("kv_format") == "auto" else "auto"),
                         evidence=f"raw/{key}/architecture/inventory.json (config sha256 {inv['config_sha256'][:12]}; "
                                  f"{inv['attention']['evidence']})",
                         status=inv["label"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    a = ap.parse_args()
    study = Path(a.study_dir)
    keys = [k for k in sc.MODEL_ORDER]
    mem, meml, dec, pre, ovd, ovp = [], [], [], [], [], []
    for key in keys:
        sel = selected(study, key)
        if not sel:
            continue
        r, lr = memory_rows(study, key, sel)
        mem += r
        meml += lr
        dec += decode_rows(study, key, sel, sel["accepted_sessions"])
        pre += prefill_rows(study, key, sel, sel["accepted_sessions"])
        ovd += decode_rows(study, key, sel, sel.get("overhead_sessions", []), probe=False)
        ovp += prefill_rows(study, key, sel, sel.get("overhead_sessions", []), probe=False)
    write_csv(study / "memory.csv", mem)
    write_csv(study / "memory_layers.csv", meml)
    write_csv(study / "decode.csv", dec)
    write_csv(study / "prefill.csv", pre)
    # overhead: matched unprobed whole-model time vs probed
    ov = []
    sd = summarize(dec, ["model", "context_tokens", "target_batch"], ["model_ms_per_step"], "decode")
    so = summarize(ovd, ["model", "context_tokens", "target_batch"], ["model_ms_per_step"], "decode")
    som = {(r["model"], r["context_tokens"], r["target_batch"]): r for r in so}
    sa = {(r["model"], r["context_tokens"], r["target_batch"]): r for r in
          summarize(dec, ["model", "context_tokens", "target_batch"], ["attention_ms_per_step"], "decode")}
    for r in sd:
        k = (r["model"], r["context_tokens"], r["target_batch"])
        u = som.get(k)
        if u and r["median"] != "" and u["median"] != "":
            delta = r["median"] - u["median"]
            att = sa.get(k, {}).get("median", "")
            ov.append(dict(experiment="decode", model=k[0], context_tokens=k[1], target_batch=k[2],
                           probed_model_ms=r["median"], unprobed_model_ms=u["median"], overhead_ms=delta,
                           overhead_pct=100 * delta / u["median"], attention_ms=att,
                           attention_lower_bound_ms=(max(att - max(delta, 0), 0) if att != "" else ""),
                           probed_valid=r["valid_repeats"], unprobed_valid=u["valid_repeats"]))
    sp = summarize(pre, ["model", "context_tokens"], ["model_ms_per_chunk"], "prefill")
    spo = {(r["model"], r["context_tokens"]): r for r in
           summarize(ovp, ["model", "context_tokens"], ["model_ms_per_chunk"], "prefill")}
    for r in sp:
        k = (r["model"], r["context_tokens"])
        u = spo.get(k)
        if u and r["median"] != "" and u["median"] != "":
            delta = r["median"] - u["median"]
            ov.append(dict(experiment="prefill", model=k[0], context_tokens=k[1], target_batch=1,
                           probed_model_ms=r["median"], unprobed_model_ms=u["median"], overhead_ms=delta,
                           overhead_pct=100 * delta / u["median"],
                           probed_valid=r["valid_repeats"], unprobed_valid=u["valid_repeats"]))
    write_csv(study / "overhead.csv", ov)
    summ = []
    summ += summarize([m for m in mem if m.get("snapshot_phase") in ("settled_decode", "")],
                      ["model", "context_tokens"], ["total_attention_state_gib", "logical_payload_gib",
                                                    "gpu_allocation_delta_gib"], "memory")
    summ += summarize(dec, ["model", "context_tokens", "target_batch"],
                      ["attention_ms_per_step", "model_ms_per_step", "effective_attention_tokens_per_s"], "decode")
    summ += summarize(pre, ["model", "context_tokens"], ["attention_ms_per_chunk", "model_ms_per_chunk"], "prefill")
    write_csv(study / "summary.csv", summ)
    arch = architecture(study, keys)
    write_csv(study / "architecture.csv", arch)
    md = ["| model | total params (B) | active / token (B) | lookup tables (B) | layers | attention types | KV/state dtype | status |",
          "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for r in arch:
        if r.get("status") == "unverified":
            md.append(f"| {r['model']} | unverified | | | | | | unverified |")
            continue
        md.append(f"| {sc.MODELS[r['model']]['display']} | {r['total_params_b']} | {r['active_decode_params_b']} | "
                  f"{r['lookup_table_params_b']} | {r['layers']} | {r['attention_types_and_counts']} | "
                  f"{r['kv_state_dtype'] or 'pending'} | {r['status']} |")
    (study / "architecture.md").write_text(
        "# Architecture inventory (calculated from checkpoint)\n\n" + "\n".join(md) +
        "\n\nActive = non-expert base weights (excluding input embedding and lookup tables) + lm_head + "
        "routed experts × experts_per_tok / n_routed. Prefill and decode use the same per-token definition. "
        "MTP/draft and vision/audio weights are excluded from active and listed in raw inventories. "
        "Packed 4-bit weights count 2 params per byte; quantization scales are excluded.\n")
    hashes = {}
    for f in sorted(study.rglob("*")):
        if f.is_file() and f.name != "manifest_hashes.json":
            hashes[str(f.relative_to(study))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (study / "manifest_hashes.json").write_text(json.dumps(hashes, indent=0))
    print(f"memory {len(mem)} rows, layers {len(meml)}, decode {len(dec)}, prefill {len(pre)}, overhead {len(ov)}, "
          f"summary {len(summ)}")


if __name__ == "__main__":
    main()

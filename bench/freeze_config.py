"""Freeze one validated deployment per model: config/<model>__selected.json (never overwritten).

Run after a smoke session passes the gate and before accepted repetitions. Records the requested and
resolved KV dtype/format, effective EP, state-buffer dtypes from the validated launch, scale evidence,
every attempted validation session and the fallback reason (if any).
  python bench/freeze_config.py --model-key K --validated-session S --kv-cache-dtype fp8 \
      --attempts S1,S2 [--fallback-reason "..."]
Accepted/overhead session ids are appended later with --add-accepted / --add-overhead (append-only).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    ap.add_argument("--validated-session")
    ap.add_argument("--kv-cache-dtype")
    ap.add_argument("--attempts", default="")
    ap.add_argument("--fallback-reason", default="")
    ap.add_argument("--add-accepted")
    ap.add_argument("--add-overhead")
    a = ap.parse_args()
    study = Path(a.study_dir)
    out = study / "config" / f"{a.model_key}__selected.json"
    if a.add_accepted or a.add_overhead:
        sel = json.loads(out.read_text())
        if a.add_accepted and a.add_accepted not in sel["accepted_sessions"]:
            sel["accepted_sessions"].append(a.add_accepted)
        if a.add_overhead and a.add_overhead not in sel["overhead_sessions"]:
            sel["overhead_sessions"].append(a.add_overhead)
        out.write_text(json.dumps(sel, indent=1))
        print(out.read_text())
        return
    if out.exists():
        raise SystemExit(f"{out} exists: a changed deployment needs a new study/config id and rerun")
    m = sc.MODELS[a.model_key]
    sdir = study / "raw" / a.model_key / "sessions" / a.validated_session
    res = json.loads((sdir / "resolved_config.json").read_text())
    info = json.loads((sdir / "probe_info.json").read_text())
    with gzip.open(sdir / "state_buffers.json.gz") as f:
        bufs = json.loads(f.read())
    r0 = [b for b in bufs if b.get("rank") == 0][0]
    kvc = json.loads((sdir / "scheduler_kv_cache_config.json").read_text())
    spec_dtypes = collections.Counter()
    formats = collections.Counter()
    for g in kvc.get("kv_cache_groups", []):
        spec = g["kv_cache_spec"]
        specs = spec.get("kv_cache_specs") or {ln: spec for ln in g["layer_names"]}
        for ln, s in specs.items():
            key = f"{s.get('__cls__')}:{s.get('dtype') or s.get('dtypes')}"
            spec_dtypes[key] += 1
            if s.get("cache_dtype_str"):
                formats[s["cache_dtype_str"]] += 1
    tensor_dtypes = collections.Counter()
    scales = collections.Counter()
    for name, v in r0["layers"].items():
        for t in v.get("tensors", []):
            tensor_dtypes[f"{v['cls'].split('.')[-1]}:{t.get('dtype')}"] += 1
        for k in ("_k_scale_float", "_v_scale_float"):
            if k in v:
                scales[f"{k}={v[k]}"] += 1
    moe = info[0].get("moe")
    ep_eff = ("N/A (dense model, no experts)" if m["dense"] else
              f"local {moe.get('local_num_experts')} of {moe.get('global_num_experts')} experts per rank"
              if moe else "unverified")
    kw = sc.engine_kwargs(a.model_key, a.kv_cache_dtype)
    cid_src = json.dumps(dict(kw=kw, env=m["env"], vllm=res["vllm"], torch=res["torch"]), sort_keys=True)
    cid = f"{a.model_key}__kv-{a.kv_cache_dtype}__" + hashlib.sha256(cid_src.encode()).hexdigest()[:10]
    sel = dict(
        config_id=cid, model=a.model_key, revision=m["revision"], validated_session=a.validated_session,
        validation_attempts=[x for x in a.attempts.split(",") if x],
        engine_kwargs=kw, env=m["env"],
        ep_requested="N/A (dense)" if m["dense"] else "on (--enable-expert-parallel)",
        ep_effective=ep_eff,
        kv_dtype_requested=a.kv_cache_dtype, kv_dtype_resolved=res["kv_cache_dtype"],
        kv_format=";".join(f"{k}x{v}" for k, v in formats.items()) or res["kv_cache_dtype"],
        state_dtype_evidence="; ".join(f"{k} x{v}" for k, v in sorted(spec_dtypes.items())) +
                             " | tensors: " + "; ".join(f"{k} x{v}" for k, v in sorted(tensor_dtypes.items())),
        kv_scale_source=("; ".join(f"{k} x{v}" for k, v in sorted(scales.items())) or "no per-layer float scales exposed")
                        + " (no calibrated scales shipped unless the checkpoint provides them)",
        fallback_reason=a.fallback_reason or "none (fp8 validated)",
        resolved=res, accepted_sessions=[], overhead_sessions=[],
    )
    out.write_text(json.dumps(sel, indent=1))
    print(json.dumps({k: sel[k] for k in ("config_id", "ep_effective", "kv_dtype_resolved", "kv_format",
                                          "state_dtype_evidence", "kv_scale_source")}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How many layers process a whole prefill chunk? (diagnostic, one rank trace)

For the last full prefill step of a trace, list every call of the kernels matching a
pattern in time order with its duration. A layer that runs on all chunk tokens shows a
long call; a layer that runs on a trimmed batch (for example V4.1's decoder layers on
each request's last 128 tokens) shows a much shorter one.

  python bench/blog_layers.py <trace.json.gz> [--pattern REGEX] [--split-ratio 0.25]
"""
import argparse, gzip, json, re, statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--pattern", default=r"humming<|marlin_moe|fused_moe", help="kernel name regex (default: expert GEMM)")
    ap.add_argument("--split-ratio", type=float, default=0.25,
                    help="calls shorter than this fraction of the longest call count as trimmed")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    gs = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context")]
    tid = max(gs, key=lambda e: e["dur"])["tid"]
    steps = []
    for e in gs:
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", e["name"])
        if e["tid"] == tid and m and int(m[2]):
            steps.append((int(m[2]), e["ts"], e["ts"] + e["dur"], e["name"]))
    full = max(s[0] for s in steps)
    _, s, t, name = [x for x in steps if x[0] == full][-1]
    calls = sorted((e["ts"], e["dur"], e["name"]) for e in ev
                   if e.get("cat") == "kernel" and s <= e["ts"] < t and re.search(a.pattern, e["name"]))
    by_name = {}
    for _, d, n in calls:
        by_name.setdefault(n[:70], []).append(d)
    out = dict(step=name, prefill_tokens=full, span_ms=(t - s) / 1e3, pattern=a.pattern, kernels={})
    for n, ds in by_name.items():
        big = [d for d in ds if d >= a.split_ratio * max(ds)]
        small = [d for d in ds if d < a.split_ratio * max(ds)]
        out["kernels"][n] = dict(calls=len(ds), total_ms=sum(ds) / 1e3,
                                 long_calls=len(big), long_median_ms=st.median(big) / 1e3,
                                 short_calls=len(small), short_median_ms=(st.median(small) / 1e3) if small else None,
                                 durations_ms_in_time_order=[round(d / 1e3, 3) for d in ds])
    if a.json:
        print(json.dumps(out))
        return
    print(f"{name}: {full} prefill tokens, span {out['span_ms']:.1f} ms")
    for n, k in out["kernels"].items():
        print(f"\n{n}\n  calls {k['calls']}, total {k['total_ms']:.1f} ms; long {k['long_calls']} "
              f"(median {k['long_median_ms']:.3f} ms); short {k['short_calls']} (median {k['short_median_ms']})")
        print("  " + " ".join(f"{d:g}" for d in k["durations_ms_in_time_order"]))


if __name__ == "__main__":
    main()

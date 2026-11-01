#!/usr/bin/env python3
"""Per-step GPU kernel breakdown from one rank's torch trace (diagnostic).
Steps are the `execute_context_*` gpu_user_annotation spans vLLM emits.
Kernel time is summed per category; categories do not overlap, but kernels on
different streams can overlap in wall time, so kernel sums can exceed the span.
Usage: bench/trace_breakdown.py <trace.json.gz> [--csv out.csv]"""
import argparse, collections, csv, gzip, json, re, statistics as st

CATS = [  # first match wins
    ("moe_expert_gemm", r"marlin_moe_wna16|fused_moe|grouped_gemm|m_grouped"),
    ("moe_route_combine", r"moe_sum|topk|moe_align|swiglu|router|gate|count_and_sort|expert"),
    ("allreduce", r"AllReduce|Allreduce|allreduce|all_reduce|oneshotAllr|twoshot"),
    ("allgather_other_comm", r"AllGather|ReduceScatter|AllToAll|nccl|nvshmem"),
    ("attention_core", r"sparse_attn_fwd|flash_fwd|FlashAttnFwd|sparse_prefill|splitkv_mla|flashmla|flash_mla"),
    ("indexer_topk", r"mqa_logits|indexer|top_k|topk_|radix|hierarch"),
    ("qkv_rope_kvcache", r"QNorm|qnorm|rope|kv_rope|insert|compress|cache"),
    ("dense_gemm", r"marlin::Marlin|nvjet|gemm|sm90_xmma|cutlass|deep_gemm"),
    ("mhc_residual_norm", r"mhc|norm"),
    ("engram", r"engram"),
]


def cat(name):
    for c, pat in CATS:
        if re.search(pat, name):
            return c
    return "other_elementwise"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--csv")
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    gspans = [e for e in ev if e.get("cat") == "gpu_user_annotation"
              and e["name"].startswith("execute_context")]
    # vLLM emits one span per CUDA stream; the main compute stream is the one
    # carrying the longest span. Kernels of all streams inside it are counted once.
    main_tid = max(gspans, key=lambda e: e["dur"])["tid"]
    spans = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in gspans
                   if e["tid"] == main_tid)
    rows = []
    for i, (s, t, name) in enumerate(spans):
        ks = [e for e in kern if s <= e["ts"] < t]
        if not ks:
            continue
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
        by = collections.Counter(); cnt = collections.Counter()
        for e in ks:
            c = cat(e["name"]); by[c] += e["dur"]; cnt[c] += 1
        tot = sum(by.values())
        row = dict(step=i, annotation=name, prefill_reqs=int(m[1]), prefill_tokens=int(m[2]),
                   decode_reqs=int(m[3]), decode_tokens=int(m[4]), span_ms=(t - s) / 1e3,
                   kernel_ms=tot / 1e3, n_kernels=len(ks))
        for c, _ in CATS + [("other_elementwise", "")]:
            row[f"{c}_ms"] = by[c] / 1e3
            row[f"{c}_n"] = cnt[c]
        rows.append(row)
    import sys
    for r in rows if len(rows) <= 12 else rows[:4] + rows[-3:]:
        kind = "PREFILL" if r["prefill_tokens"] else "DECODE"
        print(f"\nstep {r['step']} {kind} {r['annotation']}: span {r['span_ms']:.2f} ms, "
              f"kernel sum {r['kernel_ms']:.2f} ms, {r['n_kernels']} kernels")
        for c, _ in CATS + [("other_elementwise", "")]:
            if r[f"{c}_ms"]:
                print(f"  {c:22s} {r[f'{c}_ms']:8.2f} ms  {100 * r[f'{c}_ms'] / r['kernel_ms']:5.1f}%  n={r[f'{c}_n']}")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

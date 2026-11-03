"""Charts from saved CSVs only: charts/memory.png, charts/decode.png, charts/prefill.png.

Run with a separate Python that has matplotlib (never install into the frozen vLLM env):
  /root/attn-study/venv-report/bin/python bench/make_report.py
Medians are drawn; whiskers show min/max of the valid repetitions. Points without 3 valid
repetitions are drawn hollow and annotated; unsupported/failed points are labelled, never zero.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_config as sc  # noqa: E402

# Reference categorical palette (dataviz skill), fixed order, assigned by entity not rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {k: SERIES[i] for i, k in enumerate(sc.MODEL_ORDER)}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CTX_LABEL = {1024: "1K", 16384: "16K", 65536: "64K", 131072: "128K"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False})


def rd(p: Path):
    return list(csv.DictReader(open(p))) if p.exists() else []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summary_index(study: Path):
    idx = {}
    for r in rd(study / "summary.csv"):
        key = (r["experiment"], r["model"], r["context_tokens"], r.get("target_batch", ""), r["metric"])
        idx[key] = r
    return idx


def disp(k):
    return sc.MODELS[k]["display"]


def memory_chart(study: Path, out: Path):
    rows = [r for r in rd(study / "memory.csv") if r.get("status") == "measured"
            and r.get("snapshot_phase") == "settled_decode"]
    unsupported = {(r["model"], r["context_tokens"]) for r in rd(study / "memory.csv") if r.get("status") == "unsupported"}
    cats = [("ordinary_kv_gib", "ordinary KV"), ("window_kv_gib", "sliding-window KV"),
            ("shared_compressed_gib", "compressed / shared KV"), ("sparse_index_tail_gib", "sparse index / tail"),
            ("recurrent_state_gib", "recurrent state")]
    models = [k for k in sc.MODEL_ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, L in zip(axes, sc.CONTEXTS):
        for yi, k in enumerate(models):
            rs = [r for r in rows if r["model"] == k and int(r["context_tokens"]) == L]
            if not rs:
                status = "unsupported" if (k, str(L)) in unsupported else "not measured"
                ax.text(0, yi, f"  {status}", va="center", color=INK2, fontsize=8)
                continue
            rs.sort(key=lambda r: float(r["total_attention_state_gib"]))
            med = rs[len(rs) // 2]  # median repetition by total
            left = 0.0
            for ci, (col, lab) in enumerate(cats):
                v = fnum(med.get(col)) or 0.0
                if v > 0:
                    ax.barh(yi, v, left=left, color=SERIES[ci], height=0.6, edgecolor=SURF, linewidth=1.5,
                            label=lab if (L == sc.CONTEXTS[0] or True) else None)
                    left += v
            ax.text(left, yi, f" {left:.2f}" + ("" if len(rs) >= 3 else f" ({len(rs)}/3)"), va="center",
                    color=INK, fontsize=8)
        ax.set_title(f"{CTX_LABEL[L]} context", loc="left")
        ax.set_xlabel("GiB, all 8 GPUs (occupied blocks)")
        ax.set_yticks(range(len(models)), [disp(k) for k in models])
        ax.invert_yaxis()
        ax.margins(x=0.25)
    h, l = axes[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    for ax in axes[1:]:
        hh, ll = ax.get_legend_handles_labels()
        for a, b in zip(hh, ll):
            seen.setdefault(b, a)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Attention-state memory for one request (settled decode, median of valid repetitions)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def decode_chart(study: Path, out: Path):
    idx = summary_index(study)
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.75])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, B in zip(axes, sc.DECODE_BATCHES):
        for k in sc.MODEL_ORDER:
            xs, ys, lo, hi, full = [], [], [], [], []
            for L in sc.CONTEXTS:
                r = idx.get(("decode", k, str(L), str(B), "attention_ms_per_step"))
                if r and fnum(r["median"]) is not None:
                    xs.append(L)
                    ys.append(float(r["median"]))
                    lo.append(float(r["median"]) - float(r["min"]))
                    hi.append(float(r["max"]) - float(r["median"]))
                    full.append(int(r["valid_repeats"]) >= 3)
            if not xs:
                continue
            c = MODEL_COLOR[k]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=c, lw=2, marker="o", ms=5, capsize=2, label=disp(k))
            for x, y, f in zip(xs, ys, full):
                if not f:
                    ax.plot(x, y, "o", mfc=SURF, mec=c, ms=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sc.CONTEXTS, [CTX_LABEL[L] for L in sc.CONTEXTS])
        ax.set_title(f"batch {B}", loc="left")
        ax.set_xlabel("context length")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("decode attention, ms / step")
    handles, labels = axes[-1].get_legend_handles_labels()
    for ax in axes[:-1]:
        h, l = ax.get_legend_handles_labels()
        for a, b in zip(h, l):
            if b not in labels:
                handles.append(a)
                labels.append(b)
    fig.legend(handles, labels, loc="upper right", ncol=6, frameon=False)
    # fraction of whole-model step at 64K batch 8
    ax = fig.add_subplot(gs[1, :2])
    names, fr = [], []
    for k in sc.MODEL_ORDER:
        a = idx.get(("decode", k, "65536", "8", "attention_ms_per_step"))
        m = idx.get(("decode", k, "65536", "8", "model_ms_per_step"))
        if a and m and fnum(a["median"]) and fnum(m["median"]):
            names.append(disp(k))
            fr.append(100 * float(a["median"]) / float(m["median"]))
    ax.barh(range(len(names)), fr, color=[MODEL_COLOR[k] for k in sc.MODEL_ORDER if disp(k) in names],
            height=0.6)
    for i, v in enumerate(fr):
        ax.text(v, i, f" {v:.0f}%", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("attention share of whole-model decode step at 64K, batch 8 (%; probed session)")
    fig.suptitle("Decode attention time per step (median, whiskers = min/max of 3 repetitions; hollow = <3 valid)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def prefill_chart(study: Path, out: Path):
    idx = summary_index(study)
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[3, 1.2]))
    ctxs = [16384, 65536, 131072]
    n = len(sc.MODEL_ORDER)
    w = 0.8 / n
    for mi, k in enumerate(sc.MODEL_ORDER):
        for ci, L in enumerate(ctxs):
            r = idx.get(("prefill", k, str(L), "", "attention_ms_per_chunk"))
            x = ci + (mi - (n - 1) / 2) * w
            if r and fnum(r["median"]) is not None:
                v = float(r["median"])
                ax.bar(x, v, width=w * 0.92, color=MODEL_COLOR[k], label=disp(k) if ci == 0 else None,
                       yerr=[[v - float(r["min"])], [float(r["max"]) - v]], capsize=2, ecolor=INK2)
            else:
                st = r["status"] if r else "n/a"
                ax.text(x, 0, st, rotation=90, ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(ctxs)), [f"8K chunk ending at {CTX_LABEL[L]}" for L in ctxs])
    ax.set_ylabel("prefill attention, ms / 8K chunk")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Prefill attention for one 8192-token chunk (median; whiskers min/max)", loc="left")
    tx.axis("off")
    lines = ["1K first chunk (1024 tokens; not", "comparable with 8K chunks)", ""]
    for k in sc.MODEL_ORDER:
        a = idx.get(("prefill", k, "1024", "", "attention_ms_per_chunk"))
        m = idx.get(("prefill", k, "1024", "", "model_ms_per_chunk"))
        av = f"{float(a['median']):.2f}" if a and fnum(a["median"]) is not None else (a["status"] if a else "n/a")
        mv = f"{float(m['median']):.2f}" if m and fnum(m["median"]) is not None else (m["status"] if m else "n/a")
        lines.append(f"{disp(k)[:24]:<24} attn {av:>7} / model {mv:>7} ms")
    tx.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default=str(sc.STUDY_DIR))
    a = ap.parse_args()
    study = Path(a.study_dir)
    (study / "charts").mkdir(exist_ok=True)
    memory_chart(study, study / "charts" / "memory.png")
    decode_chart(study, study / "charts" / "decode.png")
    prefill_chart(study, study / "charts" / "prefill.png")
    print("charts written")


if __name__ == "__main__":
    main()
